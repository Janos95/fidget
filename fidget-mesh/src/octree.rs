//! An octree data structure and implementation of Manifold Dual Contouring

use super::{
    Mesh, Settings,
    builder::MeshBuilder,
    cell::{Cell, CellIndex, CellVertex, Leaf},
    codegen::CELL_TO_VERT_TO_EDGES,
    frame::Frame,
    qef::QuadraticErrorSolver,
    types::{Axis, CellMask, Corner, Edge},
};
use fidget_core::{
    eval::Function,
    render::{RenderHandle, RenderHints, ThreadPool},
    shape::{
        BoundShape, Shape, ShapeBulkEval, ShapeBulkEvalError, ShapeTracingEval,
        ShapeTracingEvalError, ShapeVars,
    },
    types::Grad,
};
use std::collections::VecDeque;

/// Octree storing occupancy and vertex positions for Manifold Dual Contouring
#[derive(Debug)]
pub struct Octree {
    pub(crate) root: Cell<3>,
    pub(crate) cells: Vec<[Cell<3>; 8]>,

    /// Cell vertices, given as positions within the cell
    ///
    /// This is indexed by cell leaf index; the exact shape depends heavily on
    /// the number of intersections and vertices within each leaf.
    pub(crate) verts: Vec<CellVertex<3>>,

    /// Unit surface normal at each of [`verts`](Self::verts)
    ///
    /// During construction, these are the field's raw gradients in the
    /// octree's frame (NaN where not known yet); [`Octree::build`] turns them
    /// into unit normals in model space, zero where the field has no direction.
    pub(crate) normals: Vec<nalgebra::Vector3<f32>>,

    /// Absolute collapse tolerance: the largest QEF error (sum of squared
    /// distances to the planes, in the octree's frame) a merged cell may
    /// have, from [`Settings::max_err`]; `None` for the relative rule
    pub(crate) max_err: Option<f64>,
}

impl Octree {
    /// Builds an octree with the root marked as invalid
    pub(crate) fn new() -> Self {
        Octree {
            root: Cell::Invalid,
            cells: vec![],
            verts: vec![],
            normals: vec![],
            max_err: None,
        }
    }

    /// An empty octree that collapses cells as `settings` say
    pub(crate) fn for_settings(settings: &Settings) -> Self {
        Octree {
            max_err: settings.max_err.map(|tol| {
                // the octree's [-1, 1] frame is scaled by the transform's
                // largest axis to model space
                let m = settings.world_to_model.fixed_view::<3, 3>(0, 0);
                let scale = m
                    .column(0)
                    .norm()
                    .max(m.column(1).norm())
                    .max(m.column(2).norm());
                (f64::from(tol) / f64::from(scale)).powi(2)
            }),
            ..Self::new()
        }
    }

    /// Builds an octree to the given depth
    ///
    /// The shape is evaluated on the region specified by `settings.bounds`.
    ///
    /// Returns `None` if processing is cancelled by the
    /// [`CancelToken`](fidget_core::render::CancelToken) in [`Settings`].
    pub fn build<F: Function + RenderHints + Clone>(
        b: &BoundShape<F, f32>,
        settings: &Settings,
    ) -> Option<Self> {
        let mut out = Self::build_inner(b, settings)?;

        // Apply the transform from [-1, +1] back to model space
        if settings.world_to_model != nalgebra::Matrix4::identity() {
            for v in &mut out.verts {
                let p: nalgebra::Point3<f32> = v.pos.into();
                let q = settings.world_to_model.transform_point(&p);
                v.pos = q.coords;
            }
            // Gradients taken in the octree's frame map back by the inverse
            // transpose of the transform's linear part
            let m = settings.world_to_model.fixed_view::<3, 3>(0, 0);
            if let Some(inv) = m.try_inverse() {
                let it = inv.transpose();
                for n in &mut out.normals {
                    *n = it * *n;
                }
            }
        }
        // Fill in the rest (cell vertices, and gradients that vanish or
        // aren't finite) by evaluating the field at the vertex itself
        let m = settings.world_to_model.fixed_view::<3, 3>(0, 0);
        let cell = m
            .column(0)
            .norm()
            .max(m.column(1).norm())
            .max(m.column(2).norm())
            * 2.0
            / (1u64 << settings.depth.min(63)) as f32;
        let missing: Vec<usize> = (0..out.normals.len())
            .filter(|&i| !usable(out.normals[i]))
            .collect();
        let points: Vec<_> =
            missing.iter().map(|&i| out.verts[i].pos).collect();
        for (i, n) in missing.into_iter().zip(normals(b, &points, 0.05 * cell))
        {
            out.normals[i] = n;
        }
        for n in &mut out.normals {
            *n = n.try_normalize(0.0).unwrap_or(nalgebra::Vector3::zeros());
        }
        Some(out)
    }

    /// Performs the inner work of evaluation
    ///
    /// # Panics
    /// If `shape` and `vars` are not compatible
    fn build_inner<F: Function + RenderHints + Clone>(
        b: &BoundShape<F, f32>,
        settings: &Settings,
    ) -> Option<Self> {
        let shape = b.shape();
        let vars = b.vars();
        if let Some(threads) = settings.threads {
            Self::build_inner_mt(shape, settings, vars, threads)
        } else {
            let mut eval = RenderHandle::new(shape.clone());
            let mut out = OctreeBuilder::new(settings, vars);
            let mut hermite = LeafHermiteData::default();
            if out.recurse(&mut eval, CellIndex::default(), &mut hermite) {
                Some(out.octree)
            } else {
                None
            }
        }
    }

    /// Multithreaded constructor
    fn build_inner_mt<F: Function + RenderHints + Clone>(
        shape: &Shape<F>,
        settings: &Settings,
        vars: &ShapeVars<f32>,
        threads: &ThreadPool,
    ) -> Option<Self> {
        let mut root = Octree::for_settings(settings);
        let mut todo = VecDeque::new();
        todo.push_back(CellIndex::<3>::default());
        let mut fixup = vec![];
        let mut hermites = vec![];

        // We want a number of tasks that's significantly larger than our thread
        // count, so that we can fully saturate all cores even if tasks take
        // different amounts of time.
        let target_count = (8usize.pow(u32::from(settings.depth)))
            .min(threads.thread_count() * 10);
        while todo.len() < target_count {
            let next = todo.pop_front().unwrap();

            // Reserve new cells for the 8x children
            let index = root.cells.len();
            root.cells.push([Cell::Invalid; 8]);
            hermites.push([LeafHermiteData::default(); 8]);
            for i in Corner::<3>::iter() {
                let cell = next.child(index, i);
                todo.push_back(cell);
            }
            fixup.push((next, index));
        }

        use rayon::prelude::*;

        struct Output {
            cell: CellIndex<3>,
            octree: Octree,
            hermite: LeafHermiteData,
        }
        let mut rh = RenderHandle::new(shape.clone());
        let _ = rh.i_tape(&mut vec![]); // pre-populate interval tape
        let out = threads.run(|| {
            todo.par_iter()
                .map_init(
                    || (OctreeBuilder::new(settings, vars), rh.clone()),
                    |(builder, eval), cell| {
                        let mut hermite = LeafHermiteData::default();
                        // Patch our cell so that it builds at index 0
                        let local_cell = CellIndex {
                            index: None,
                            ..*cell
                        };
                        if !builder.recurse(eval, local_cell, &mut hermite) {
                            return None;
                        }
                        let octree = std::mem::replace(
                            &mut builder.octree,
                            Octree::for_settings(settings),
                        );
                        Some(Output {
                            octree,
                            cell: *cell,
                            hermite,
                        })
                    },
                )
                .collect::<Option<Vec<_>>>()
        })?;

        // Copy hermite data into arrays, and compute cumulative offsets
        let mut cell_offsets = vec![root.cells.len()];
        let mut vert_offsets = vec![0];
        for o in &out {
            let (i, j) = o.cell.index.unwrap();
            hermites[i][j as usize] = o.hermite;
            let c = cell_offsets.last().unwrap() + o.octree.cells.len();
            cell_offsets.push(c);
            let v = vert_offsets.last().unwrap() + o.octree.verts.len();
            vert_offsets.push(v);
        }
        root.cells.reserve(*cell_offsets.last().unwrap());
        root.verts.reserve(*vert_offsets.last().unwrap());
        root.normals.reserve(*vert_offsets.last().unwrap());

        for (i, o) in out.into_iter().enumerate() {
            assert_eq!(cell_offsets[i], root.cells.len());
            assert_eq!(vert_offsets[i], root.verts.len());
            let remap_cell = |c| match c {
                Cell::Leaf(Leaf { mask, index }) => Cell::Leaf(Leaf {
                    mask,
                    index: index + vert_offsets[i],
                }),
                Cell::Branch { index } => Cell::Branch {
                    index: index + cell_offsets[i],
                },
                Cell::Full | Cell::Empty => c,
                Cell::Invalid => panic!(),
            };
            root.cells.extend(
                o.octree.cells.into_iter().map(|cs| cs.map(remap_cell)),
            );
            root.verts.extend(o.octree.verts);
            root.normals.extend(o.octree.normals);
            root[o.cell] = remap_cell(o.octree.root);
        }

        // Walk back up the tree, merging cells as we go
        for (cell, index) in fixup.into_iter().rev() {
            let h = hermites[index];
            root[cell] = root.check_done(
                cell,
                index,
                h,
                cell.index
                    .map(|(i, j)| &mut hermites[i][j as usize])
                    .unwrap_or(&mut LeafHermiteData::default()),
            );
        }
        Some(root)
    }

    /// Recursively walks the dual of the octree, building a mesh
    pub fn walk_dual(&self) -> Mesh {
        let mut mesh = MeshBuilder::default();

        mesh.cell(self, CellIndex::default());
        mesh.take()
    }

    pub(crate) fn is_leaf(&self, cell: CellIndex<3>) -> bool {
        match self[cell] {
            Cell::Leaf(..) | Cell::Full | Cell::Empty => true,
            Cell::Branch { .. } => false,
            Cell::Invalid => panic!(),
        }
    }

    /// Looks up the given child of a cell.
    ///
    /// If the cell is a leaf node, returns that cell instead.
    ///
    /// # Panics
    /// If the cell is [`Invalid`](Cell::Invalid)
    pub(crate) fn child<C: Into<Corner<3>>>(
        &self,
        cell: CellIndex<3>,
        child: C,
    ) -> CellIndex<3> {
        let child = child.into();

        match self[cell] {
            Cell::Leaf { .. } | Cell::Full | Cell::Empty => cell,
            Cell::Branch { index } => cell.child(index, child),
            Cell::Invalid => panic!(),
        }
    }

    /// Checks the set of 8 children starting at the given index for completion
    ///
    /// If all are empty or full, then pro-actively collapses the cells (freeing
    /// them if they're at the tail end of the array).
    fn check_done(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_data: [LeafHermiteData; 8],
        hermite: &mut LeafHermiteData, // output
    ) -> Cell<3> {
        // Check out the children
        let mut full_count = 0;
        let mut empty_count = 0;
        for i in 0..8 {
            match self.cells[index][i] {
                Cell::Invalid => {
                    panic!("found invalid cell during meshing")
                }
                Cell::Full => {
                    full_count += 1;
                }
                Cell::Empty => {
                    empty_count += 1;
                }
                Cell::Branch { .. } => return Cell::Branch { index },
                Cell::Leaf(Leaf { .. }) => (),
            }
        }

        // If all of the branches are empty or full, then we're going to
        // record an Empty or Full cell in the parent and don't need the
        // 8x children.  Drop them by resizing the array
        let out = if full_count == 8 {
            Cell::Full
        } else if empty_count == 8 {
            Cell::Empty
        } else if let Some(leaf) =
            self.try_collapse(cell, index, hermite_data, hermite)
        {
            Cell::Leaf(leaf)
        } else {
            Cell::Branch { index }
        };

        // If we can collapse the cell, then we'll be recording an Empty / Full
        // / Leaf cell in the parent and don't need the 8x children.
        //
        // We can drop them if they happen to be at the tail end of the
        // octree; otherwise, we'll satisfy ourselves with setting them to
        // invalid.  We will always be at the tail end of the octree during
        // single-threaded evaluation, it's only during merging octrees from
        // multiple threads that we'd have cells midway through the array.
        if !matches!(out, Cell::Branch { .. }) {
            if index == self.cells.len() - 1 {
                self.cells.resize(index, [Cell::Invalid; 8]);
            } else {
                self.cells[index] = [Cell::Invalid; 8];
            }
        }
        out
    }

    /// Try to collapse the given cell
    ///
    /// Writes to `self.verts` and returns the leaf if successful; otherwise
    /// returns `None`.
    fn try_collapse(
        &mut self,
        cell: CellIndex<3>,
        index: usize,
        hermite_data: [LeafHermiteData; 8],
        hermite: &mut LeafHermiteData, // output
    ) -> Option<Leaf<3>> {
        let mask = self.collapsible(index)?;
        *hermite = LeafHermiteData::merge(hermite_data)?;

        // Empty / full cells should never be produced here.  The
        // only way to get an empty / full cell is for all eight
        // corners to be empty / full; if that was the case, then
        // either:
        //
        // - The interior vertices match, in which case this should
        //   have been collapsed into a single empty / full cell
        // - The interior vertices *do not* match, in which case the
        //   cell should not be marked as collapsible.
        let (pos, new_err) = hermite.solve();
        let reject = match self.max_err {
            // an absolute tolerance: the same in model units whatever the
            // octree's size (the relative rule's error floor scales with it)
            Some(max_err) => hermite.solve_exact().1 > max_err,
            None => new_err >= hermite.qef_err * 2.0,
        };
        if reject || !cell.bounds.contains(pos) {
            return None;
        }
        // Record the newly-collapsed leaf
        hermite.qef_err = new_err;

        let index = self.verts.len();
        self.verts.push(pos);
        self.normals.push(nalgebra::Vector3::repeat(f32::NAN));

        // Install cell intersections, of which there must only
        // be one (since we only collapse manifold cells)
        let edges = &CELL_TO_VERT_TO_EDGES[mask.index()];
        debug_assert_eq!(edges.len(), 1);
        for e in edges[0] {
            let i = hermite.intersections[e.to_undirected().index()];
            self.verts.push(CellVertex { pos: i.pos.xyz() });
            self.normals.push(i.grad.xyz());
        }

        Some(Leaf { mask, index })
    }

    /// Checks whether the set of 8 cells beginning at `root` can be collapsed.
    ///
    /// Only topology is checked, based on the three predicates from "Dual
    /// Contouring of Hermite Data" (Ju et al, 2002), §4.1
    pub(crate) fn collapsible(&self, root: usize) -> Option<CellMask<3>> {
        // Copy cells to a local variable for simplicity
        let cells = self.cells[root];

        let mut mask = 0;
        for (i, &c) in cells.iter().enumerate() {
            let b = match c {
                Cell::Leaf(Leaf { mask, .. }) => {
                    if CELL_TO_VERT_TO_EDGES[mask.index()].len() > 1 {
                        return None;
                    }
                    (mask.index() & (1 << i) != 0) as u8
                }
                Cell::Empty => 0,
                Cell::Full => 1,
                Cell::Branch { .. } => return None,
                Cell::Invalid => panic!(),
            };
            mask |= b << i;
        }

        use super::frame::{XYZ, YZX, ZXY};
        for (t, u, v) in [XYZ::frame(), YZX::frame(), ZXY::frame()] {
            //  - The sign in the middle of a coarse edge must agree with the
            //    sign of at least one of the edge’s two endpoints.
            for i in 0..4 {
                let a = (u * ((i & 1) != 0)) | (v * ((i & 2) != 0));
                let b = a | t;
                let center = cells[a.index()].corner(b);

                if [a, b]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return None;
                }
            }

            //  - The sign in the middle of a coarse face must agree with the
            //    sign of at least one of the face’s four corners.
            for i in 0..2 {
                let a: Corner<3> = (t * (i & 1 == 0)).into();
                let b = a | u;
                let c = a | v;
                let d = a | u | v;

                let center = cells[a.index()].corner(d);

                if [a, b, c, d]
                    .iter()
                    .all(|v| ((mask & (1 << v.index())) != 0) != center)
                {
                    return None;
                }
            }
            //  - The sign in the middle of a coarse cube must agree with the
            //    sign of at least one of the cube’s eight corners.
            for _i in 0..1 {
                // Doing this in the t,u,v loop isn't strictly necessary, but it
                // preserves the structure nicely.
                let center = cells[0].corner(t | u | v);
                if (0..8).all(|v| ((mask & (1 << v)) != 0) != center) {
                    return None;
                }
            }
        }

        // The outer cell must not be empty or full at this point; if it was
        // empty or full and the other conditions had been met, then it should
        // have been collapsed already.
        debug_assert_ne!(mask, 255);
        debug_assert_ne!(mask, 0);

        // The merged cell gets one vertex, so it must have one vertex's worth
        // of surface: a single disc (§3.2 of Schaefer, Ju and Warren's
        // "Manifold Dual Contouring", 2007).  The signs above don't see
        // everything inside: a tube through the cell, or two sheets meeting at
        // one corner, would pinch the mesh at the merged vertex.
        if CELL_TO_VERT_TO_EDGES[mask as usize].len() == 1 && is_disc(&cells) {
            Some(CellMask::new(mask))
        } else {
            None
        }
    }
}

impl std::ops::Index<CellIndex<3>> for Octree {
    type Output = Cell<3>;

    fn index(&self, i: CellIndex<3>) -> &Self::Output {
        match i.index {
            None => &self.root,
            Some((i, j)) => &self.cells[i][j as usize],
        }
    }
}

impl std::ops::IndexMut<CellIndex<3>> for Octree {
    fn index_mut(&mut self, i: CellIndex<3>) -> &mut Self::Output {
        match i.index {
            None => &mut self.root,
            Some((i, j)) => &mut self.cells[i][j as usize],
        }
    }
}

/// Data structure for an under-construction octree
pub(crate) struct OctreeBuilder<'a, F: Function + RenderHints> {
    /// In-construction octree
    pub(crate) octree: Octree,

    // Values stolen from settings
    cancel: fidget_core::render::CancelToken,
    world_to_model: Option<&'a nalgebra::Matrix4<f32>>,
    max_depth: u8,
    vars: &'a ShapeVars<f32>,

    eval_float_slice: ShapeBulkEval<F::FloatSliceEval>,
    eval_interval: ShapeTracingEval<F::IntervalEval>,
    eval_grad_slice: ShapeBulkEval<F::GradSliceEval>,

    tape_storage: Vec<F::TapeStorage>,
    shape_storage: Vec<F::Storage>,
    workspace: F::Workspace,
}

impl<'a, F: Function + RenderHints> OctreeBuilder<'a, F> {
    /// Builds a new octree builder, which allocates data for 8 root cells
    pub(crate) fn new(
        settings: &'a Settings,
        vars: &'a ShapeVars<f32>,
    ) -> Self {
        let world_to_model =
            if settings.world_to_model == nalgebra::Matrix4::identity() {
                None
            } else {
                Some(&settings.world_to_model)
            };
        Self {
            octree: Octree::for_settings(settings),

            cancel: settings.cancel.clone(),
            max_depth: settings.depth,
            world_to_model,
            vars,

            eval_float_slice: Shape::<F>::new_float_slice_eval(),
            eval_grad_slice: Shape::<F>::new_grad_slice_eval(),
            eval_interval: Shape::<F>::new_interval_eval(),
            tape_storage: vec![],
            shape_storage: vec![],
            workspace: Default::default(),
        }
    }

    /// Recurse down the octree, building the given cell
    ///
    /// Writes to `self.o.cells[cell]`, which must be reserved
    ///
    /// If a leaf is written, then `hermite` is populated
    ///
    /// Returns `true` if recursion completed normally, or `false` if it was
    /// cancelled.
    #[must_use]
    fn recurse(
        &mut self,
        eval: &mut RenderHandle<F>,
        cell: CellIndex<3>,
        hermite: &mut LeafHermiteData,
    ) -> bool {
        if self.cancel.is_cancelled() {
            return false;
        }
        let (i, r) = match self.eval_interval.eval_raw(
            eval.i_tape(&mut self.tape_storage),
            cell.bounds[crate::types::X],
            cell.bounds[crate::types::Y],
            cell.bounds[crate::types::Z],
            self.world_to_model,
            self.vars,
        ) {
            Ok(v) => v,
            Err(ShapeTracingEvalError::MissingVar(..)) => unreachable!(),
        };
        self.octree[cell] = if i.upper() < 0.0 {
            Cell::Full
        } else if i.lower() > 0.0 {
            Cell::Empty
        } else {
            let sub_tape = if F::simplify_tree_during_meshing(cell.depth) {
                if let Some(trace) = r.as_ref() {
                    eval.simplify(
                        trace,
                        &mut self.workspace,
                        &mut self.shape_storage,
                        &mut self.tape_storage,
                    )
                } else {
                    eval
                }
            } else {
                eval
            };
            if cell.depth == self.max_depth as usize {
                self.leaf(sub_tape, cell, hermite)
            } else {
                // Reserve new cells for the 8x children
                let index = self.octree.cells.len();
                self.octree.cells.push([Cell::Invalid; 8]);
                let mut hermite_child = [LeafHermiteData::default(); 8];
                for i in Corner::<3>::iter() {
                    let cell = cell.child(index, i);
                    if !self.recurse(
                        sub_tape,
                        cell,
                        &mut hermite_child[i.index()],
                    ) {
                        return false;
                    }
                }

                // Figure out whether the children can be collapsed
                self.octree.check_done(cell, index, hermite_child, hermite)
            }
        };
        true
    }

    /// Evaluates the given leaf
    ///
    /// Writes the leaf vertex to `self.o.verts`, hermite data to
    /// `self.hermite`, and the leaf data to `self.leafs`.  Does **not** write
    /// anything to `self.o.cells`; the cell is returned instead.
    fn leaf(
        &mut self,
        eval: &mut RenderHandle<F>,
        cell: CellIndex<3>,
        hermite_cell: &mut LeafHermiteData,
    ) -> Cell<3> {
        let mut xs = [0.0; 8];
        let mut ys = [0.0; 8];
        let mut zs = [0.0; 8];
        for i in Corner::<3>::iter() {
            let [x, y, z] = cell.corner(i);
            xs[i.index()] = x;
            ys[i.index()] = y;
            zs[i.index()] = z;
        }

        let out = match self.eval_float_slice.eval_raw(
            eval.f_tape(&mut self.tape_storage),
            &xs,
            &ys,
            &zs,
            self.world_to_model,
            ShapeBulkEval::<F::FloatSliceEval>::var_value(self.vars),
        ) {
            Ok(v) => v,
            Err(
                ShapeBulkEvalError::MissingVar(..)
                | ShapeBulkEvalError::MismatchedVarSlices { .. },
            ) => unreachable!(),
        };
        debug_assert_eq!(out.len(), 8);

        // Build a mask of active corners, which determines cell
        // topology / vertex count / active edges / etc.
        let mask = out
            .iter()
            .enumerate()
            .filter(|(_i, v)| **v < 0.0)
            .fold(0, |acc, (i, _v)| acc | (1 << i));

        // How fast the field changes across the cell: a gradient far below
        // this is no gradient (see `flat` below)
        let slope = {
            let (lo, hi) = out
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
            (hi - lo)
                / (cell.corner(Corner::new(7))[0]
                    - cell.corner(Corner::new(0))[0])
        };

        // Early exit if the cell is completely empty or full
        if mask == 0 {
            return Cell::Empty;
        } else if mask == 255 {
            return Cell::Full;
        }

        let mask = CellMask::new(mask);

        // Start and endpoints in 3D space for intersection searches
        let mut start = [nalgebra::Vector3::zeros(); 12];
        let mut end = [nalgebra::Vector3::zeros(); 12];
        let mut edge_count = 0;

        // Convert from the corner mask to start and end-points in 3D space,
        // relative to the cell bounds (i.e. as a `Matrix3<u16>`).
        for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
            for e in *vs {
                // Find the axis that's being used by this edge
                let axis = e.start().index() ^ e.end().index();
                debug_assert_eq!(axis.count_ones(), 1);
                debug_assert!(axis < 8);

                // Pick a position closer to the filled side
                let (a, b) = if e.end().index() & axis != 0 {
                    (0, u16::MAX)
                } else {
                    (u16::MAX, 0)
                };

                // Convert the intersection to a 3D position
                let mut v = nalgebra::Vector3::zeros();
                let i = (axis.trailing_zeros() + 1) % 3;
                let j = (axis.trailing_zeros() + 2) % 3;
                v[i as usize] = if e.start() & Axis::new(1 << i) {
                    u16::MAX
                } else {
                    0
                };
                v[j as usize] = if e.start() & Axis::new(1 << j) {
                    u16::MAX
                } else {
                    0
                };

                v[axis.trailing_zeros() as usize] = a;
                start[edge_count] = v;
                v[axis.trailing_zeros() as usize] = b;
                end[edge_count] = v;
                edge_count += 1;
            }
        }
        // Slice off the unused sections of the arrays
        let start = &mut start[..edge_count]; // always inside
        let end = &mut end[..edge_count]; // always outside

        // Scratch arrays for edge search
        const EDGE_SEARCH_SIZE: usize = 16;
        const EDGE_SEARCH_DEPTH: usize = 4;
        let xs =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let ys =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];
        let zs =
            &mut [0.0; 12 * EDGE_SEARCH_SIZE][..edge_count * EDGE_SEARCH_SIZE];

        // The inside corner of each edge, kept for gradients below
        let inside: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 12> =
            start.iter().cloned().collect();

        // This part looks hairy, but it's just doing an N-ary search along each
        // edge to find the intersection point.
        for _ in 0..EDGE_SEARCH_DEPTH {
            // Populate edge search arrays
            let mut i = 0;
            for (start, end) in start.iter().zip(end.iter()) {
                for j in 0..EDGE_SEARCH_SIZE {
                    let pos = ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - j - 1) as u32)
                        + (end.map(|i| i as u32) * j as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32);
                    debug_assert!(pos.max() <= u16::MAX.into());

                    let pos = cell.pos(pos.map(|p| p as u16));
                    xs[i] = pos.x;
                    ys[i] = pos.y;
                    zs[i] = pos.z;
                    i += 1;
                }
            }
            debug_assert_eq!(i, EDGE_SEARCH_SIZE * edge_count);

            // Do the actual evaluation
            let out = match self.eval_float_slice.eval_raw(
                eval.f_tape(&mut self.tape_storage),
                xs,
                ys,
                zs,
                self.world_to_model,
                ShapeBulkEval::<F::FloatSliceEval>::var_value(self.vars),
            ) {
                Ok(v) => v,
                Err(
                    ShapeBulkEvalError::MissingVar(..)
                    | ShapeBulkEvalError::MismatchedVarSlices { .. },
                ) => unreachable!(),
            };

            // Update start and end positions based on evaluation
            for ((start, end), search) in start
                .iter_mut()
                .zip(end.iter_mut())
                .zip(out.chunks(EDGE_SEARCH_SIZE))
            {
                // The search must be inside-to-outside
                debug_assert!(search[0] < 0.0);
                debug_assert!(search[EDGE_SEARCH_SIZE - 1] >= 0.0);
                let frac = search
                    .iter()
                    .enumerate()
                    .find(|(_i, v)| **v >= 0.0)
                    .unwrap()
                    .0;
                debug_assert!(frac > 0);
                debug_assert!(frac < EDGE_SEARCH_SIZE);

                let f = |frac| {
                    ((start.map(|i| i as u32)
                        * (EDGE_SEARCH_SIZE - frac - 1) as u32)
                        + (end.map(|i| i as u32) * frac as u32))
                        / ((EDGE_SEARCH_SIZE - 1) as u32)
                };

                let a = f(frac - 1);
                let b = f(frac);

                debug_assert!(a.max() <= u16::MAX.into());
                debug_assert!(b.max() <= u16::MAX.into());
                *start = a.map(|v| v as u16);
                *end = b.map(|v| v as u16);
            }
        }

        // Populate intersections to the average of start and end
        let intersections: arrayvec::ArrayVec<nalgebra::Vector3<u16>, 12> =
            start
                .iter()
                .zip(end.iter())
                .map(|(a, b)| {
                    ((a.map(|v| v as u32) + b.map(|v| v as u32)) / 2)
                        .map(|v| v as u16)
                })
                .collect();

        let xs = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];
        let ys = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];
        let zs = &mut [Grad::from(0.0); 12 * EDGE_SEARCH_SIZE]
            [..intersections.len()];

        for (i, xyz) in intersections.iter().enumerate() {
            let pos = cell.pos(*xyz);
            xs[i] = Grad::new(pos.x, 1.0, 0.0, 0.0);
            ys[i] = Grad::new(pos.y, 0.0, 1.0, 0.0);
            zs[i] = Grad::new(pos.z, 0.0, 0.0, 1.0);
        }

        // TODO: special case for cells with multiple gradients ("features")
        let grads = match self.eval_grad_slice.eval_raw(
            eval.g_tape(&mut self.tape_storage),
            xs,
            ys,
            zs,
            self.world_to_model,
            ShapeBulkEval::<F::GradSliceEval>::var_value(self.vars),
        ) {
            Ok(v) => v,
            Err(
                ShapeBulkEvalError::MissingVar(..)
                | ShapeBulkEvalError::MismatchedVarSlices { .. },
            ) => unreachable!(),
        };
        let mut grads: arrayvec::ArrayVec<Grad, 12> =
            grads.iter().cloned().collect();

        // An exact distance field can have no gradient exactly on its surface
        // (e.g. the derivative of `sqrt(x² + y²)` where both are zero, the
        // outside of a box); where the search lands there, the intersection
        // would add no plane to the QEF and the vertex would slide off a sharp
        // edge, or two cells' vertices would land on one point.  It does land
        // there on faces aligned with the grid, and far from the origin, where
        // the search's last steps round to the same coordinates (and the
        // gradient there is rounding noise: a thousandth of the field's slope
        // across the cell counts as none).  Take the gradient a 64th of the
        // cell inside along the edge instead.
        let flat: arrayvec::ArrayVec<usize, 12> = (0..grads.len())
            .filter(|&i| {
                let g = grads[i];
                [g.dx, g.dy, g.dz].iter().all(|f| f.is_finite())
                    && g.dx.abs().max(g.dy.abs()).max(g.dz.abs())
                        <= 1e-3 * slope
            })
            .collect();
        if !flat.is_empty() {
            let mut fx = [Grad::from(0.0); 12];
            let mut fy = [Grad::from(0.0); 12];
            let mut fz = [Grad::from(0.0); 12];
            for (k, &i) in flat.iter().enumerate() {
                let p = start[i].map(|v| v as i32);
                let q = inside[i].map(|v| v as i32);
                let step = (q - p).map(|d| d.clamp(-1024, 1024));
                let pos = cell.pos((p + step).map(|v| v as u16));
                fx[k] = Grad::new(pos.x, 1.0, 0.0, 0.0);
                fy[k] = Grad::new(pos.y, 0.0, 1.0, 0.0);
                fz[k] = Grad::new(pos.z, 0.0, 0.0, 1.0);
            }
            let n = flat.len();
            let inside = match self.eval_grad_slice.eval_raw(
                eval.g_tape(&mut self.tape_storage),
                &fx[..n],
                &fy[..n],
                &fz[..n],
                self.world_to_model,
                ShapeBulkEval::<F::GradSliceEval>::var_value(self.vars),
            ) {
                Ok(v) => v,
                Err(
                    ShapeBulkEvalError::MissingVar(..)
                    | ShapeBulkEvalError::MismatchedVarSlices { .. },
                ) => unreachable!(),
            };
            for (&i, g) in flat.iter().zip(inside) {
                grads[i] = Grad::new(grads[i].v, g.dx, g.dy, g.dz);
            }
        }

        let mut verts: arrayvec::ArrayVec<_, 4> = arrayvec::ArrayVec::new();
        let mut i = 0;
        for vs in CELL_TO_VERT_TO_EDGES[mask.index()].iter() {
            let mut force_point = None;
            let mut qef = QuadraticErrorSolver::new();
            for e in vs.iter() {
                let pos = nalgebra::Vector3::new(xs[i].v, ys[i].v, zs[i].v);
                let grad: nalgebra::Vector4<f32> = grads[i].into();

                // If a point has invalid gradients, then it's _probably_ on a
                // sharp feature in the mesh, so we should just snap to that
                // point specifically.  This means we don't solve the QEF, and
                // instead mark it as invalid.
                if grad.iter().any(|f| f.is_nan()) {
                    force_point = Some(pos);
                    hermite_cell.qef_err = QEF_ERR_INVALID;
                    break;
                }
                qef.add_intersection(pos, grad);

                // Record this intersection in the Hermite data for the leaf
                let edge_index = e.to_undirected().index();
                hermite_cell.intersections[edge_index] = LeafIntersection {
                    pos: nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0),
                    grad,
                };

                i += 1;
            }

            if let Some(pos) = force_point {
                verts.push(CellVertex { pos });
            } else {
                let (pos, err) = qef.solve();
                verts.push(pos);

                // We overwrite the error here, because it's only used when
                // collapsing cells, which only occurs if there's a single
                // vertex; last-error-wins works fine in that case.
                hermite_cell.qef_err = err;
            }
        }

        let index = self.octree.verts.len();
        self.octree.normals.extend(
            std::iter::repeat_n(
                nalgebra::Vector3::repeat(f32::NAN),
                verts.len(),
            )
            .chain(
                grads
                    .iter()
                    .map(|g| nalgebra::Vector3::new(g.dx, g.dy, g.dz)),
            ),
        );
        self.octree.verts.extend(verts);
        self.octree.verts.extend(
            intersections
                .into_iter()
                .map(|pos| CellVertex { pos: cell.pos(pos) }),
        );

        Cell::Leaf(Leaf { mask, index })
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Whether the surface inside eight sibling cells (each with at most one
/// vertex) is one disc
///
/// The siblings' corners form a 3×3×3 grid of signs.  The surface cuts that
/// grid's edges at points, its faces along arcs and its cells in discs (one
/// per cell with a vertex), so its Euler characteristic is points - arcs +
/// discs; a connected surface with a boundary is a disc iff that is 1.
fn is_disc(cells: &[Cell<3>; 8]) -> bool {
    // Sign at grid point (x, y, z), each 0..=2
    let inside = |x: usize, y: usize, z: usize| -> bool {
        let (cx, cy, cz) = (x.min(1), y.min(1), z.min(1));
        let child = cx | (cy << 1) | (cz << 2);
        let corner = (x - cx) | ((y - cy) << 1) | ((z - cz) << 2);
        cells[child].corner(Corner::new(corner as u8))
    };
    let mut points = 0;
    for axis in 0..3 {
        for a in 0..2 {
            for b in 0..3 {
                for c in 0..3 {
                    let p = |t: usize| match axis {
                        0 => inside(t, b, c),
                        1 => inside(c, t, b),
                        _ => inside(b, c, t),
                    };
                    points += (p(a) != p(a + 1)) as i32;
                }
            }
        }
    }
    // Faces: a square with corners at (u, v) .. (u + 1, v + 1) in the plane
    // normal to `axis` at `w`; it has one arc per two crossings.  Faces
    // inside the 2×2×2 block (w == 1) also join the discs on either side.
    let mut arcs = 0;
    let mut parent = [0, 1, 2, 3, 4, 5, 6, 7];
    fn find(p: &mut [usize; 8], i: usize) -> usize {
        let mut i = i;
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for axis in 0..3 {
        for w in 0..3 {
            for u in 0..2 {
                for v in 0..2 {
                    let at = |du: usize, dv: usize| match axis {
                        0 => inside(w, u + du, v + dv),
                        1 => inside(v + dv, w, u + du),
                        _ => inside(u + du, v + dv, w),
                    };
                    let ring = [at(0, 0), at(1, 0), at(1, 1), at(0, 1)];
                    let crossings = (0..4)
                        .filter(|&k| ring[k] != ring[(k + 1) % 4])
                        .count();
                    arcs += crossings as i32 / 2;
                    if w == 1 && crossings > 0 {
                        let cell = |side: usize| -> usize {
                            let (x, y, z) = match axis {
                                0 => (side, u, v),
                                1 => (v, side, u),
                                _ => (u, v, side),
                            };
                            x | (y << 1) | (z << 2)
                        };
                        let (a, b) = (
                            find(&mut parent, cell(0)),
                            find(&mut parent, cell(1)),
                        );
                        parent[a] = b;
                    }
                }
            }
        }
    }
    let with_vertex: Vec<usize> = (0..8)
        .filter(|&i| matches!(cells[i], Cell::Leaf(..)))
        .collect();
    let discs = with_vertex.len() as i32;
    let connected = with_vertex
        .iter()
        .all(|&i| find(&mut parent, i) == find(&mut parent, with_vertex[0]));
    connected && points - arcs + discs == 1
}

/// Whether a gradient gives a direction
fn usable(g: nalgebra::Vector3<f32>) -> bool {
    g.iter().all(|f| f.is_finite()) && g.amax() > 1e-12
}

/// The field's gradient at each point (model space, not normalized)
///
/// Where the gradient doesn't give a direction (it isn't finite, or it
/// vanishes, as the derivative of a distance does on a sharp edge or exactly
/// on a polygon's side), it's estimated by central differences with the given
/// step instead; where that vanishes too, the result is zero.
pub fn normals<F: Function + RenderHints + Clone>(
    b: &BoundShape<F, f32>,
    points: &[nalgebra::Vector3<f32>],
    step: f32,
) -> Vec<nalgebra::Vector3<f32>> {
    const CHUNK: usize = 4096;
    let (shape, vars) = (b.shape(), b.vars());
    let mut out = Vec::with_capacity(points.len());
    if points.is_empty() {
        return out;
    }
    let tape = shape.grad_slice_tape(Default::default());
    let mut eval = Shape::<F>::new_grad_slice_eval();
    let (mut xs, mut ys, mut zs) = (vec![], vec![], vec![]);
    for chunk in points.chunks(CHUNK) {
        xs.clear();
        ys.clear();
        zs.clear();
        for p in chunk {
            xs.push(Grad::new(p.x, 1.0, 0.0, 0.0));
            ys.push(Grad::new(p.y, 0.0, 1.0, 0.0));
            zs.push(Grad::new(p.z, 0.0, 0.0, 1.0));
        }
        let r = eval
            .eval_raw(
                &tape,
                &xs,
                &ys,
                &zs,
                None,
                ShapeBulkEval::<F::GradSliceEval>::var_value(vars),
            )
            .expect("shape and vars are compatible");
        out.extend(r.iter().map(|g| nalgebra::Vector3::new(g.dx, g.dy, g.dz)));
    }
    let bad: Vec<usize> = (0..out.len()).filter(|&i| !usable(out[i])).collect();
    if !bad.is_empty() {
        let tape = shape.float_slice_tape(Default::default());
        let mut eval = Shape::<F>::new_float_slice_eval();
        let (mut xs, mut ys, mut zs) = (vec![], vec![], vec![]);
        for chunk in bad.chunks(CHUNK) {
            xs.clear();
            ys.clear();
            zs.clear();
            for &i in chunk {
                for axis in 0..3 {
                    for s in [step, -step] {
                        let mut q = points[i];
                        q[axis] += s;
                        xs.push(q.x);
                        ys.push(q.y);
                        zs.push(q.z);
                    }
                }
            }
            let r = eval
                .eval_raw(
                    &tape,
                    &xs,
                    &ys,
                    &zs,
                    None,
                    ShapeBulkEval::<F::FloatSliceEval>::var_value(vars),
                )
                .expect("shape and vars are compatible");
            for (k, &i) in chunk.iter().enumerate() {
                let d = &r[6 * k..6 * k + 6];
                let g = nalgebra::Vector3::new(
                    d[0] - d[1],
                    d[2] - d[3],
                    d[4] - d[5],
                );
                out[i] = if usable(g) {
                    g
                } else {
                    nalgebra::Vector3::zeros()
                };
            }
        }
    }
    out
}

#[derive(Copy, Clone, Default, Debug)]
struct LeafIntersection {
    /// Intersection position is xyz; w is 1 if the intersection is present
    pos: nalgebra::Vector4<f32>,
    /// Gradient is xyz; w is the distance field value at this point
    grad: nalgebra::Vector4<f32>,
}

impl From<LeafIntersection> for QuadraticErrorSolver {
    fn from(i: LeafIntersection) -> Self {
        let mut qef = QuadraticErrorSolver::default();
        if i.pos.w != 0.0 {
            qef.add_intersection(i.pos.xyz(), i.grad);
        }
        qef
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct LeafHermiteData {
    intersections: [LeafIntersection; 12],
    face_qefs: [QuadraticErrorSolver; 6],
    center_qef: QuadraticErrorSolver,

    /// Error found when solving this QEF (if positive), or a special value
    qef_err: f32,
}

/// This QEF is not populated
const QEF_ERR_EMPTY: f32 = -1.0;

/// This QEF is known to be invalid and should be disregarded
const QEF_ERR_INVALID: f32 = -2.0;

impl Default for LeafHermiteData {
    fn default() -> Self {
        Self {
            intersections: Default::default(),
            face_qefs: Default::default(),
            center_qef: Default::default(),
            qef_err: QEF_ERR_EMPTY,
        }
    }
}

impl LeafHermiteData {
    /// Merges an octree subdivision of leaf hermite data
    ///
    /// Returns `None` if any of the leafs have invalid QEFs (typically due to
    /// NANs in normal computation).
    fn merge(leafs: [LeafHermiteData; 8]) -> Option<Self> {
        let mut out = Self::default();
        use super::types::{X, Y, Z};

        if leafs.iter().any(|v| v.qef_err == QEF_ERR_INVALID) {
            return None;
        }

        // Accumulate intersections along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = u.next();
            for edge in 0..4 {
                let mut start = Corner::<3>::new(0);
                if edge & 1 != 0 {
                    start = start | u;
                }
                if edge & 2 != 0 {
                    start = start | v;
                }
                let end = start | t;

                // Canonical edge as a value in the 0-12 range
                let edge = Edge::new((t.index() * 4 + edge) as u8);

                // One or the other leaf has an intersection
                let a = leafs[start.index()].intersections[edge.index()];
                let b = leafs[end.index()].intersections[edge.index()];
                match (a.pos.w > 0.0, b.pos.w > 0.0) {
                    (true, false) => out.intersections[edge.index()] = a,
                    (false, true) => out.intersections[edge.index()] = b,
                    (false, false) => (),
                    (true, true) => panic!("duplicate intersection"),
                }
            }
        }

        // Accumulate face QEFs along edges
        for t in [X, Y, Z] {
            let u = t.next();
            let v = u.next();
            for face in 0..2 {
                let a = if face == 1 {
                    t.into()
                } else {
                    Corner::<3>::new(0)
                };
                let b = a | u;
                let c = a | v;
                let d = a | u | v;
                let f = t.index() * 2 + face;
                for q in [a, b, c, d] {
                    out.face_qefs[f] += leafs[q.index()].face_qefs[f];
                }
                // The face's inner edges: along u at v = mid (halves in
                // children a and b; an edge along u is indexed by its offsets
                // along v then t) ...
                let along_u = u.index() * 4 + 1 + face * 2;
                out.face_qefs[f] +=
                    leafs[a.index()].intersections[along_u].into();
                out.face_qefs[f] +=
                    leafs[b.index()].intersections[along_u].into();

                // ... and along v at u = mid (halves in children a and c; an
                // edge along v is indexed by its offsets along t then u)
                let along_v = v.index() * 4 + face + 2;
                out.face_qefs[f] +=
                    leafs[a.index()].intersections[along_v].into();
                out.face_qefs[f] +=
                    leafs[c.index()].intersections[along_v].into();
            }
        }

        // Accumulate center QEFs
        for t in [X, Y, Z] {
            let u = t.next();
            let v = u.next();

            // Accumulate the four inner face QEFs
            let a = Corner::<3>::new(0);
            let b = a | u;
            let c = a | v;
            let d = a | u | v;
            for q in [a, b, c, d] {
                out.center_qef += leafs[q.index()].face_qefs[t.index() * 2 + 1];
            }

            // Edges oriented along the u axis
            out.center_qef +=
                leafs[a.index()].intersections[u.index() * 4 + 3].into();
            out.center_qef +=
                leafs[b.index()].intersections[u.index() * 4 + 3].into();

            // We skip edges oriented on the v axis, because they'll be counted
            // by one of the other iterations through the loop
        }
        for leaf in leafs {
            out.center_qef += leaf.center_qef;
        }

        // Accumulate minimum QEF error among valid child QEFs
        out.qef_err = f32::INFINITY;
        for e in leafs.iter().map(|q| q.qef_err).filter(|&e| e >= 0.0) {
            out.qef_err = out.qef_err.min(e);
        }

        Some(out)
    }

    /// The combined QEF
    fn qef(&self) -> QuadraticErrorSolver {
        let mut qef = self.center_qef;
        for &i in &self.intersections {
            qef += i.into();
        }
        for &f in &self.face_qefs {
            qef += f;
        }
        qef
    }

    /// Solves the combined QEF
    pub fn solve(&self) -> (CellVertex<3>, f32) {
        self.qef().solve()
    }

    /// Solves the combined QEF, with the error unclamped
    pub fn solve_exact(&self) -> (CellVertex<3>, f64) {
        self.qef().solve_exact()
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::types::{Edge, X, Y, Z};
    use fidget_core::{
        context::{Context, Tree},
        render::ThreadPool,
        shape::EzShape,
        var::Var,
        vm::{VmFunction, VmShape},
    };
    use std::collections::BTreeMap;

    fn depth0_single_thread() -> Settings<'static> {
        Settings {
            depth: 0,
            threads: None,
            ..Default::default()
        }
    }

    fn depth1_single_thread() -> Settings<'static> {
        Settings {
            depth: 1,
            threads: None,
            ..Default::default()
        }
    }

    /// Converts a [`Tree`] with no extra variables to a [`BoundShape`]
    ///
    /// # Panics
    /// If the tree uses variables
    fn tree_to_shape(t: Tree) -> BoundShape<'static, VmFunction, f32> {
        VmShape::from(t).try_into().expect("no variables allowed")
    }

    fn sphere(center: [f32; 3], radius: f32) -> Tree {
        let (x, y, z) = Tree::axes();
        ((x - center[0]).square()
            + (y - center[1]).square()
            + (z - center[2]).square())
        .sqrt()
            - radius
    }

    fn cube(bx: [f32; 2], by: [f32; 2], bz: [f32; 2]) -> Tree {
        let (x, y, z) = Tree::axes();
        let x_bounds = (bx[0] - x.clone()).max(x - bx[1]);
        let y_bounds = (by[0] - y.clone()).max(y - by[1]);
        let z_bounds = (bz[0] - z.clone()).max(z - bz[1]);
        x_bounds.max(y_bounds).max(z_bounds)
    }

    #[test]
    fn test_cube_edge() {
        const EPSILON: f32 = 1e-3;
        let f = 2.0;
        let shape = tree_to_shape(cube([-f, f], [-f, 0.3], [-f, 0.6]));
        // This should be a cube with a single edge running through the root
        // node of the octree, with an edge vertex at [0, 0.3, 0.6]
        let octree = Octree::build(&shape, &depth0_single_thread()).unwrap();
        assert_eq!(octree.verts.len(), 5);
        let v = octree.verts[0].pos;
        let expected = nalgebra::Vector3::new(0.0, 0.3, 0.6);
        assert!(
            (v - expected).norm() < EPSILON,
            "bad edge vertex {v:?}; expected {expected:?}"
        );
    }

    fn cone(
        corner: nalgebra::Vector3<f32>,
        tip: nalgebra::Vector3<f32>,
        radius: f32,
    ) -> Tree {
        let dir = tip - corner;
        let length = dir.norm();
        let dir = dir.normalize();

        let corner = corner.map(Tree::constant);
        let dir = dir.map(Tree::constant);

        let (x, y, z) = Tree::axes();
        let point = nalgebra::Vector3::new(x, y, z);
        let offset = point.clone() - corner.clone();

        // a is the distance along the corner-tip direction
        let a = offset.x.clone() * dir.x.clone()
            + offset.y.clone() * dir.y.clone()
            + offset.z.clone() * dir.z.clone();

        // Position of the nearest point on the corner-tip axis
        let a_pos = corner + dir * a.clone();

        // b is the orthogonal distance
        let offset = point - a_pos;
        let b = (offset.x.clone().square()
            + offset.y.clone().square()
            + offset.z.clone().square())
        .sqrt();

        b - radius * (1.0 - a / length)
    }

    #[test]
    fn test_mesh_basic() {
        let shape = tree_to_shape(sphere([0.0; 3], 0.2));

        // If we only build a depth-0 octree, then it's a leaf without any
        // vertices (since all the corners are empty)
        let octree = Octree::build(&shape, &depth0_single_thread()).unwrap();
        assert!(octree.cells.is_empty()); // root only
        assert_eq!(Cell::Empty, octree.root);
        assert!(octree.verts.is_empty());

        let empty_mesh = octree.walk_dual();
        assert!(empty_mesh.vertices.is_empty());
        assert!(empty_mesh.triangles.is_empty());

        // Now, at depth-1, each cell should be a Leaf with one vertex
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert_eq!(octree.cells.len(), 1); // one fanout of 8 cells
        assert_eq!(Cell::Branch { index: 0 }, octree.root);

        // Each of the 6 edges is counted 4 times and each cell has 1 vertex
        assert_eq!(octree.verts.len(), 6 * 4 + 8, "incorrect vertex count");

        // Each cell is a leaf with 4 vertices (3 edges, 1 center)
        for o in octree.cells[0].iter() {
            let Cell::Leaf(Leaf { index, mask }) = *o else {
                panic!()
            };
            assert_eq!(mask.count_ones(), 1);
            assert_eq!(index % 4, 0);
        }

        let sphere_mesh = octree.walk_dual();
        assert!(sphere_mesh.vertices.len() > 1);
        assert!(!sphere_mesh.triangles.is_empty());
    }

    #[test]
    fn test_sphere_verts() {
        let shape = tree_to_shape(sphere([0.0; 3], 0.2));

        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        let sphere_mesh = octree.walk_dual();

        let mut edge_count = 0;
        let mut corner_count = 0;
        for v in &sphere_mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);
            if edge_sum == 1 {
                assert!(
                    (v.norm() - 0.2).abs() < 2.0 / u16::MAX as f32,
                    "edge vertex {v:?} is not at radius 0.2"
                );
                edge_count += 1;
            } else {
                // The sphere looks like a box at this sampling depth, since the
                // edge intersections are all axis-aligned; we'll check that
                // corner vertices are all at [±0.2, ±0.2, ±0.2]
                assert!(
                    (v.abs() - nalgebra::Vector3::new(0.2, 0.2, 0.2)).norm()
                        < 2.0 / 65535.0,
                    "cell vertex {v:?} is not at expected position"
                );
                corner_count += 1;
            }
        }
        // Each quad around an edge is flat (a face of the box), so it is split
        // by a diagonal and the edge vertices aren't used: a box of 12 triangles
        assert_eq!(edge_count, 0);
        assert_eq!(corner_count, 8);
        assert_eq!(sphere_mesh.triangles.len(), 12);
    }

    #[test]
    fn test_sphere_manifold() {
        let shape = tree_to_shape(sphere([0.0; 3], 0.85));

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            let sphere_mesh = octree.walk_dual();

            check_for_vertex_dupes(&sphere_mesh).unwrap();
            check_for_edge_matching(&sphere_mesh).unwrap();
        }
    }

    /// A torus tilted against the grid, so cells of different sizes meet
    /// along its curved surface
    fn tilted_torus() -> Tree {
        let (x, y, z) = Tree::axes();
        let (c, s) = (0.4f32.cos(), 0.4f32.sin());
        let y2 = y.clone() * c - z.clone() * s;
        let z2 = y * s + z * c;
        let ring = (x.square() + y2.square()).sqrt() - 0.55;
        (ring.square() + z2.square()).sqrt() - 0.2
    }

    #[test]
    fn test_vertex_normals() {
        let shape = tree_to_shape(sphere([0.1, 0.0, -0.05], 0.6));
        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let mesh = Octree::build(&shape, &settings).unwrap().walk_dual();
            assert_eq!(mesh.normals.len(), mesh.vertices.len());
            for (p, n) in mesh.vertices.iter().zip(&mesh.normals) {
                let radial =
                    (p - nalgebra::Vector3::new(0.1, 0.0, -0.05)).normalize();
                assert!(
                    (n.norm() - 1.0).abs() < 1e-4,
                    "{n:?} is not a unit normal"
                );
                assert!(n.dot(&radial) > 0.9999, "normal {n:?} at {p:?}");
            }
        }
    }

    #[test]
    fn test_no_folds_where_cell_sizes_change() {
        // Fanning around the edge's intersection folds ~2% of the area here
        let shape = tree_to_shape(tilted_torus());
        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 7,
                threads,
                ..Default::default()
            };
            let mesh = Octree::build(&shape, &settings).unwrap().walk_dual();
            check_for_vertex_dupes(&mesh).unwrap();
            check_for_edge_matching(&mesh).unwrap();
            let (mut total, mut folded) = (0.0, 0.0);
            for t in &mesh.triangles {
                let [a, b, c] = [t.x, t.y, t.z].map(|i| mesh.vertices[i]);
                let n = (b - a).cross(&(c - a));
                let r =
                    mesh.normals[t.x] + mesh.normals[t.y] + mesh.normals[t.z];
                total += n.norm();
                if n.dot(&r) < 0.0 {
                    folded += n.norm();
                }
            }
            assert!(
                folded / total < 1e-3,
                "{:.3}% of the area folds",
                100.0 * folded / total
            );
        }
    }

    #[test]
    fn test_no_zero_area_where_the_surface_grazes_a_corner() {
        // The corner at the origin is inside by less than one step of the edge
        // search, so the cell above it has a cap too small to see: its vertex
        // and three intersections all land on the corner, and fanning around
        // those edges would make triangles with no area
        let (x, y, z) = Tree::axes();
        let ball =
            (x.clone().square() + y.clone().square() + z.clone().square())
                .sqrt()
                - 0.9;
        let shape = tree_to_shape((x + y + z - 1e-7).max(ball));
        for depth in [4, 5, 6] {
            let settings = Settings {
                depth,
                threads: None,
                ..Default::default()
            };
            let mesh = Octree::build(&shape, &settings).unwrap().walk_dual();
            check_for_edge_matching(&mesh).unwrap();
            check_for_manifold_vertices(&mesh).unwrap();
        }
    }

    #[test]
    fn test_two_caps_are_two_vertices() {
        // Six inside corners around two opposite outside ones (0 and 7): one
        // inside region, but the surface is two caps, one around each outside
        // corner, and one vertex for both would pinch the mesh
        assert_eq!(CELL_TO_VERT_TO_EDGES[0b0111_1110].len(), 2);
        assert_eq!(CELL_TO_VERT_TO_EDGES[0b1000_0001].len(), 2);
    }

    /// A union of capsules (rods with round ends) at pseudo-random places,
    /// thin enough that the cells see their crossings, tangencies and gaps
    fn rods(seed: u64, count: usize) -> Tree {
        let mut state =
            seed.wrapping_mul(6364136223846793005) ^ 0x9e3779b97f4a7c15;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) as f32
        };
        let (x, y, z) = Tree::axes();
        let mut out: Option<Tree> = None;
        for _ in 0..count {
            let a = [0; 3].map(|_| 1.6 * next() - 0.8);
            let b = [0; 3].map(|_| 1.6 * next() - 0.8);
            let r = 0.03 + 0.09 * next();
            let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let len2 = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
            let pa = [x.clone() - a[0], y.clone() - a[1], z.clone() - a[2]];
            let h = ((pa[0].clone() * ba[0]
                + pa[1].clone() * ba[1]
                + pa[2].clone() * ba[2])
                / len2)
                .max(0.0)
                .min(1.0);
            let d = ((pa[0].clone() - h.clone() * ba[0]).square()
                + (pa[1].clone() - h.clone() * ba[1]).square()
                + (pa[2].clone() - h * ba[2]).square())
            .sqrt()
                - r;
            out = Some(match out {
                None => d,
                Some(o) => o.min(d),
            });
        }
        out.unwrap()
    }

    /// Checks that every vertex's triangles form one fan (its link is a
    /// single cycle: no pinch), and that no triangle has zero area
    fn check_for_manifold_vertices(mesh: &Mesh) -> Result<(), String> {
        let mut next: Vec<BTreeMap<usize, usize>> =
            vec![BTreeMap::new(); mesh.vertices.len()];
        for t in &mesh.triangles {
            let [a, b, c] = [t.x, t.y, t.z];
            for (v, p, q) in [(a, b, c), (b, c, a), (c, a, b)] {
                if next[v].insert(p, q).is_some() {
                    return Err(format!("vertex {v} repeats a triangle edge"));
                }
            }
            let [pa, pb, pc] = [a, b, c].map(|i| mesh.vertices[i]);
            if (pb - pa).cross(&(pc - pa)).norm() == 0.0 {
                return Err(format!("zero-area triangle at {pa:?}"));
            }
        }
        for (v, link) in next.iter().enumerate() {
            let Some((&start, _)) = link.iter().next() else {
                continue;
            };
            let (mut cur, mut steps) = (start, 0);
            loop {
                cur = *link
                    .get(&cur)
                    .ok_or_else(|| format!("open fan at vertex {v}"))?;
                steps += 1;
                if cur == start {
                    break;
                }
            }
            if steps != link.len() {
                return Err(format!(
                    "pinched vertex {v} at {:?}",
                    mesh.vertices[v]
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn test_meshes_are_closed_manifolds() {
        // Thin rods cross, touch and nearly touch: cells with two patches
        // (rods 13 at depth 5), ambiguous faces whose arcs both cells join (a
        // tunnel through the face; also rods 13), and merged cells whose
        // surface isn't one disc
        // (six rods, seed 1: a merged cell around a tube at depth 6)
        for (seed, count) in (0..40).map(|s| (s, 4)).chain([(1, 6)]) {
            let shape = tree_to_shape(rods(seed, count));
            for depth in [5, 6, 7] {
                let settings = Settings {
                    depth,
                    threads: None,
                    ..Default::default()
                };
                let mesh =
                    Octree::build(&shape, &settings).unwrap().walk_dual();
                let ok = check_for_edge_matching(&mesh)
                    .and_then(|_| check_for_manifold_vertices(&mesh));
                if let Err(e) = ok {
                    panic!("rods {seed} at depth {depth}: {e}");
                }
            }
        }
    }

    #[test]
    fn test_exact_box_keeps_its_edges() {
        // An exact distance (`length(max(q, 0)) + min(max(q), 0)`) has no
        // gradient on its faces' grid-aligned points; the vertices of merged
        // cells must still land on the box's edges, not slide off them
        let (x, y, z) = Tree::axes();
        let qx = (x - 0.125).abs() - 0.5;
        let qy = (y + 0.0625).abs() - 0.375;
        let qz = z.abs() - 0.25;
        let outside = (qx.clone().max(0.0).square()
            + qy.clone().max(0.0).square()
            + qz.clone().max(0.0).square())
        .sqrt();
        let inside = qx.max(qy).max(qz).min(0.0);
        let shape = tree_to_shape(outside + inside);
        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 6,
                threads,
                ..Default::default()
            };
            let mesh = Octree::build(&shape, &settings).unwrap().walk_dual();
            let mut volume = 0.0;
            for t in &mesh.triangles {
                let [a, b, c] = [t.x, t.y, t.z].map(|i| mesh.vertices[i]);
                volume += a.dot(&b.cross(&c)) / 6.0;
            }
            let expected = 1.0 * 0.75 * 0.5;
            assert!(
                (volume.abs() - expected).abs() < 1e-4,
                "volume {volume} for {expected}"
            );
            for p in &mesh.vertices {
                let d = nalgebra::Vector3::new(
                    (p.x - 0.125).abs() - 0.5,
                    (p.y + 0.0625).abs() - 0.375,
                    p.z.abs() - 0.25,
                );
                assert!(d.max().abs() < 1e-4, "vertex {p:?} is off the box");
            }
        }
    }

    #[test]
    fn test_cube_verts() {
        let shape = tree_to_shape(cube([-0.1, 0.6], [-0.2, 0.75], [-0.3, 0.4]));

        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        let mesh = octree.walk_dual();
        const EPSILON: f32 = 2.0 / u16::MAX as f32;
        assert!(!mesh.vertices.is_empty());
        for v in &mesh.vertices {
            // Edge vertices should be found via binary search and therefore
            // should be close to the true crossing point
            let x_edge = v.x != 0.0;
            let y_edge = v.y != 0.0;
            let z_edge = v.z != 0.0;
            let edge_sum = x_edge as u8 + y_edge as u8 + z_edge as u8;
            assert!(edge_sum == 1 || edge_sum == 3);

            if edge_sum == 1 {
                assert!(
                    (x_edge
                        && ((v.x - -0.1).abs() < EPSILON
                            || (v.x - 0.6).abs() < EPSILON))
                        || (y_edge
                            && ((v.y - -0.2).abs() < EPSILON
                                || (v.y - 0.75).abs() < EPSILON))
                        || (z_edge
                            && ((v.z - -0.3).abs() < EPSILON
                                || (v.z - 0.4).abs() < EPSILON)),
                    "bad edge position {v:?}"
                );
            } else {
                assert!(
                    ((v.x - -0.1).abs() < EPSILON
                        || (v.x - 0.6).abs() < EPSILON)
                        && ((v.y - -0.2).abs() < EPSILON
                            || (v.y - 0.75).abs() < EPSILON)
                        && ((v.z - -0.3).abs() < EPSILON
                            || (v.z - 0.4).abs() < EPSILON),
                    "bad vertex position {v:?}"
                );
            }
        }
    }

    #[test]
    fn test_plane_center() {
        const EPSILON: f32 = 1e-3;
        for dx in [0.0, 0.25, -0.25, 2.0, -2.0] {
            for dy in [0.0, 0.25, -0.25, 2.0, -2.0] {
                for offset in [0.0, -0.2, 0.2] {
                    let (x, y, z) = Tree::axes();
                    let f = x * dx + y * dy + z + offset;
                    let shape = tree_to_shape(f);
                    let octree =
                        Octree::build(&shape, &depth0_single_thread()).unwrap();

                    assert!(octree.cells.is_empty()); // root only
                    let pos = octree.verts[0].pos;
                    let mut mass_point = nalgebra::Vector3::zeros();
                    for v in &octree.verts[1..] {
                        mass_point += v.pos;
                    }
                    mass_point /= (octree.verts.len() - 1) as f32;
                    assert!(
                        (pos - mass_point).norm() < EPSILON,
                        "bad vertex position at dx: {dx}, dy: {dy}, \
                         offset: {offset} => {pos:?} != {mass_point:?}"
                    );
                    let mut eval = VmShape::new_point_eval();
                    let tape = shape.shape().ez_point_tape();
                    for v in &octree.verts {
                        let v = v.pos;
                        let (r, _) = eval.eval(&tape, v.x, v.y, v.z).unwrap();
                        assert!(r.abs() < EPSILON, "bad value at {v:?}: {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_cone_vert() {
        // Test both in-cell and out-of-cell cone vertices
        for tip in [
            nalgebra::Vector3::new(0.2, 0.3, 0.4),
            nalgebra::Vector3::new(1.2, 1.3, 1.4),
        ] {
            let corner = nalgebra::Vector3::new(-1.0, -1.0, -1.0);
            let shape = tree_to_shape(cone(corner, tip, 0.1));

            let mut eval = VmShape::new_point_eval();
            let tape = shape.shape().ez_point_tape();
            let (v, _) = eval.eval(&tape, tip.x, tip.y, tip.z).unwrap();
            assert!(v.abs() < 1e-6, "bad tip value: {v}");
            let (v, _) =
                eval.eval(&tape, corner.x, corner.y, corner.z).unwrap();
            assert!(v < 0.0, "bad corner value: {v}");

            let octree =
                Octree::build(&shape, &depth0_single_thread()).unwrap();
            assert!(octree.cells.is_empty()); // root only
            assert_eq!(octree.verts.len(), 4);

            let pos = octree.verts[0].pos;
            assert!(
                (pos - tip).norm() < 1e-3,
                "bad vertex position: expected {tip:?}, got {pos:?}"
            );
        }
    }

    fn test_mesh_manifold_inner(threads: Option<&ThreadPool>, mask: u8) {
        let mut shape = vec![];
        for j in Corner::<3>::iter() {
            if mask & (1 << j.index()) != 0 {
                shape.push(sphere(
                    [
                        if j & X { 0.5 } else { 0.0 },
                        if j & Y { 0.5 } else { 0.0 },
                        if j & Z { 0.5 } else { 0.0 },
                    ],
                    0.1,
                ));
            }
        }
        let Some(start) = shape.pop() else { return };
        let shape = shape.into_iter().fold(start, |acc, s| acc.min(s));

        // Now, we have our shape, which is 0-8 spheres placed at the
        // corners of the cell spanning [0, 0.25]
        let shape = tree_to_shape(shape);
        let settings = Settings {
            depth: 2,
            threads,
            ..Default::default()
        };
        let octree = Octree::build(&shape, &settings).unwrap();

        let mesh = octree.walk_dual();
        if mask != 0 && mask != 255 {
            assert!(!mesh.vertices.is_empty());
            assert!(!mesh.triangles.is_empty());
        }

        if let Err(e) = check_for_vertex_dupes(&mesh) {
            panic!("mask {mask:08b} has {e}");
        }
        if let Err(e) = check_for_edge_matching(&mesh) {
            panic!("mask {mask:08b} has {e}");
        }
    }

    #[test]
    fn test_mesh_manifold_single_thread() {
        for mask in 0..=255 {
            test_mesh_manifold_inner(None, mask)
        }
    }

    #[test]
    fn test_mesh_manifold_multi_thread() {
        for mask in 0..=255 {
            test_mesh_manifold_inner(Some(&ThreadPool::Global), mask)
        }
    }

    #[test]
    fn test_collapsible() {
        let shape = tree_to_shape(sphere([0.0; 3], 0.1));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(octree.collapsible(0).is_none());

        // If we have a single corner sphere, then the builder will collapse the
        // branch for us, leaving just a leaf.
        let shape = tree_to_shape(sphere([-1.0; 3], 0.1));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(matches!(octree.root, Cell::Leaf { .. }));

        // Test with a single thread and a deeper tree, which should still work
        let octree = Octree::build(
            &shape,
            &Settings {
                depth: 4,
                threads: None,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            matches!(octree.root, Cell::Leaf { .. }),
            "root should be a leaf, not {:?}",
            octree.root
        );

        // This should also work with multiple threads!
        let octree = Octree::build(
            &shape,
            &Settings {
                depth: 4,
                threads: Some(&ThreadPool::Global),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            matches!(octree.root, Cell::Leaf { .. }),
            "root should be a leaf, not {:?}",
            octree.root
        );

        let shape = tree_to_shape(sphere([-1.0, 0.0, 1.0], 0.1));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(octree.collapsible(0).is_none());

        let a = sphere([-1.0; 3], 0.1);
        let b = sphere([1.0; 3], 0.1);
        let shape = tree_to_shape(a.min(b));
        let octree = Octree::build(&shape, &depth1_single_thread()).unwrap();
        assert!(octree.collapsible(0).is_none());
    }

    #[test]
    fn test_empty_collapse() {
        // Make a very smol sphere that won't be sampled
        let shape = tree_to_shape(sphere([0.1; 3], 0.05));

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 1,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            assert_eq!(
                octree.root,
                Cell::Empty,
                "failed to collapse octree with threads: {:?}",
                threads.map(|t| t.thread_count())
            );
        }
    }

    #[test]
    fn test_colonnade_manifold() {
        const COLONNADE: &str = include_str!("../../models/colonnade.vm");
        let (ctx, root) = Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();
        let shape = tape.try_into().unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            let mesh = octree.walk_dual();
            // Note: the model has duplicate vertices!
            if let Err(e) = check_for_edge_matching(&mesh) {
                panic!(
                    "colonnade model has {e} with threads: {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    #[test]
    fn colonnade_bounds() {
        const COLONNADE: &str = include_str!("../../models/colonnade.vm");
        let (ctx, root) = Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();
        let shape = tape.try_into().unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 8,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            let mesh = octree.walk_dual();
            for v in mesh.vertices.iter() {
                assert!(
                    v.x < 1.0
                        && v.x > -1.0
                        && v.y < 1.0
                        && v.y > -1.0
                        && v.z < 1.0
                        && v.z > -0.5,
                    "invalid vertex {v:?} with threads {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    #[test]
    fn bear_bounds() {
        const COLONNADE: &str = include_str!("../../models/bear.vm");
        let (ctx, root) = Context::from_text(COLONNADE.as_bytes()).unwrap();
        let tape = VmShape::new(&ctx, root).unwrap();
        let shape = tape.try_into().unwrap();

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 5,
                threads,
                ..Default::default()
            };
            let octree = Octree::build(&shape, &settings).unwrap();
            let mesh = octree.walk_dual();
            for v in mesh.vertices.iter() {
                assert!(
                    v.x < 1.0
                        && v.x > -0.75
                        && v.y < 1.0
                        && v.y > -0.75
                        && v.z < 0.75
                        && v.z > -0.75,
                    "invalid vertex {v:?} with threads {:?}",
                    threads.map(|t| t.thread_count())
                );
            }
        }
    }

    fn check_for_vertex_dupes(mesh: &Mesh) -> Result<(), String> {
        let mut verts = mesh.vertices.clone();
        verts.sort_by_key(|k| (k.x.to_bits(), k.y.to_bits(), k.z.to_bits()));
        for i in 1..verts.len() {
            if verts[i - 1] == verts[i] {
                return Err(format!("duplicate vertices at {}", verts[i]));
            }
        }
        Ok(())
    }

    fn check_for_edge_matching(mesh: &Mesh) -> Result<(), String> {
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                if t.x == t.y || t.y == t.z || t.x == t.z {
                    return Err("triangle with duplicate edges".to_string());
                }
                *edges.entry(edge).or_default() += 1;
            }
        }
        for (&(a, b), &i) in &edges {
            if i != 1 {
                return Err(format!(
                    "duplicate edge ({a}, {b}) between {:?} {:?}",
                    mesh.vertices[a], mesh.vertices[b]
                ));
            }
            if !edges.contains_key(&(b, a)) {
                return Err("unpaired edges".to_owned());
            }
        }
        Ok(())
    }

    #[test]
    fn test_qef_merging() {
        let mut hermite = LeafHermiteData::default();

        // Add a dummy intersection with a non-zero normal; we'll be counting
        // mass points to check that the merging went smoothly
        let grad = nalgebra::Vector4::new(1.0, 0.0, 0.0, 0.0);
        let pos = nalgebra::Vector4::new(0.0, 0.0, 0.0, 1.0);
        hermite.intersections.fill(LeafIntersection { pos, grad });

        // Ensure that only one of the two sub-edges per edge contains an
        // intersection, because that's checked for in `LeafHermiteData::merge`
        let mut hermites = [hermite; 8];
        for i in 0..12 {
            let e = Edge::new(i);
            let (_start, end) = e.corners();
            hermites[end.index()].intersections[i as usize] =
                LeafIntersection::default();
        }

        let merged = LeafHermiteData::merge(hermites).unwrap();
        for i in merged.intersections {
            assert_eq!(i.grad, grad);
            assert_eq!(i.pos, pos);
        }
        // Each face in the merged cell should include the accumulation of four
        // edges from lower cells (but nothing more, because lower cells didn't
        // have face QEFs populated)
        for (i, f) in merged.face_qefs.iter().enumerate() {
            assert_eq!(
                f.mass_point().w,
                4.0,
                "bad accumulated QEF on face {i}"
            );
        }
        assert_eq!(
            merged.center_qef.mass_point().w,
            6.0,
            "bad accumulated QEF in center"
        );
    }

    #[test]
    fn test_qef_near_planar() {
        let shape = tree_to_shape(sphere([0.0; 3], 0.75));

        let settings = Settings {
            depth: 4,
            ..Default::default()
        };

        let octree = Octree::build(&shape, &settings).unwrap().walk_dual();
        for v in octree.vertices.iter() {
            let n = v.norm();
            assert!(n > 0.7 && n < 0.8, "invalid vertex at {v:?}: {n}");
        }
    }

    #[test]
    fn test_mesh_vars() {
        let (x, y, z) = Tree::axes();
        let v = Var::new();
        let c = Tree::from(v);
        let sphere = (x.square() + y.square() + z.square()).sqrt() - c;
        let shape = VmShape::from(sphere);

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 4,
                threads,
                ..Default::default()
            };

            for r in [0.5, 0.75] {
                let mut vars = ShapeVars::new();
                vars.insert(v.index().unwrap(), r);
                let shape = shape.bind(&vars).unwrap();
                let octree =
                    Octree::build(&shape, &settings).unwrap().walk_dual();
                for v in octree.vertices.iter() {
                    let n = v.norm();
                    assert!(
                        n > r - 0.05 && n < r + 0.05,
                        "invalid vertex at {v:?}: {n} != {r} with threads {:?}",
                        threads.map(|t| t.thread_count())
                    );
                }
            }
        }
    }

    #[test]
    fn test_octree_cancel() {
        let (x, y, z) = Tree::axes();
        let sphere = (x.square() + y.square() + z.square()).sqrt() - 1.0;
        let shape = tree_to_shape(sphere);

        for threads in [None, Some(&ThreadPool::Global)] {
            let settings = Settings {
                depth: 4,
                threads,
                ..Default::default()
            };
            let c = settings.cancel.clone();
            c.cancel();

            assert!(Octree::build(&shape, &settings).is_none());
        }
    }

    fn cylinder(r: f32, h: f32) -> Tree {
        let (x, y, z) = Tree::axes();
        ((x.square() + y.square()).sqrt() - r).max(z.abs() - h)
    }

    fn count_bad_edges(mesh: &Mesh) -> (usize, usize) {
        // (edges not used exactly once each way, plus vertices that aren't
        // finite or sit exactly on another vertex; triangles that are
        // degenerate or doubled once vertices are identified by position)
        let mut edges: BTreeMap<_, usize> = BTreeMap::new();
        for t in &mesh.triangles {
            for edge in [(t.x, t.y), (t.y, t.z), (t.z, t.x)] {
                *edges.entry(edge).or_default() += 1;
            }
        }
        let bad_edges = edges
            .iter()
            .filter(|&(&(a, b), &n)| n != 1 || edges.get(&(b, a)) != Some(&1))
            .count();
        let mut at: BTreeMap<_, usize> = BTreeMap::new();
        let mut bad_verts = 0;
        let id: Vec<usize> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| {
                bad_verts += !v.iter().all(|c| c.is_finite()) as usize;
                let k = (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
                let n = at.len();
                let j = *at.entry(k).or_insert(i);
                bad_verts += (j != i) as usize;
                let _ = n;
                j
            })
            .collect();
        let mut tris: BTreeMap<_, usize> = BTreeMap::new();
        let mut bad_tris = 0;
        for t in &mesh.triangles {
            let mut k = [id[t.x], id[t.y], id[t.z]];
            k.sort();
            bad_tris += (k[0] == k[1] || k[1] == k[2]) as usize;
            *tris.entry(k).or_default() += 1;
        }
        bad_tris += tris.values().filter(|&&n| n > 1).count();
        (bad_edges + bad_verts, bad_tris)
    }

    fn mesh_report(name: &str, t: Tree, depth: u8) -> (usize, usize) {
        let shape = tree_to_shape(t);
        let settings = Settings {
            depth,
            threads: None,
            ..Default::default()
        };
        let mesh = Octree::build(&shape, &settings).unwrap().walk_dual();
        let r = count_bad_edges(&mesh);
        eprintln!(
            "{name} depth {depth}: {} triangles, {} bad edges or vertices, {} degenerate or doubled triangles",
            mesh.triangles.len(),
            r.0,
            r.1
        );
        r
    }

    #[test]
    fn test_tule_reproducers() {
        let mut fails = 0;
        for depth in [5, 6, 7] {
            for (name, t) in [
                ("cylinder", cylinder(0.5, 0.6)),
                ("thin cylinder", cylinder(0.31, 0.77)),
                ("exact cylinder", {
                    // as written for an exact distance field: sqrt(max(d, 0)²)
                    // inside, which used to give NaN gradients
                    let (x, y, z) = Tree::axes();
                    let dr = (x.square() + y.square()).sqrt() - 0.5;
                    let dz = z.abs() - 0.6;
                    (dr.clone().max(0.0).square()
                        + dz.clone().max(0.0).square())
                    .sqrt()
                        + dr.max(dz).min(0.0)
                }),
                ("exact box", {
                    let (x, y, z) = Tree::axes();
                    let (dx, dy, dz) =
                        (x.abs() - 0.5, y.abs() - 0.4, z.abs() - 0.3);
                    (dx.clone().max(0.0).square()
                        + dy.clone().max(0.0).square()
                        + dz.clone().max(0.0).square())
                    .sqrt()
                        + dx.max(dy).max(dz).min(0.0)
                }),
                (
                    "touching spheres",
                    sphere([-0.3, 0.0, 0.0], 0.3)
                        .min(sphere([0.3, 0.0, 0.0], 0.3)),
                ),
                ("knife edge", {
                    let (x, y, z) = Tree::axes();
                    // a wedge thinning to 0.002 at |y| = 0.8: like a chine or a wing edge
                    (z.abs() - (0.8 - y.abs()) * 0.05 - 0.002)
                        .max(y.abs() - 0.8)
                        .max(x.abs() - 0.7)
                }),
            ] {
                let (b, p) = mesh_report(name, t, depth);
                fails += (b > 0 || p > 0) as usize;
            }
        }
        assert_eq!(fails, 0);
    }

    /// Merging eight leaves counts each intersection on the parent's inner
    /// edges, and each child's face QEF, exactly once
    #[test]
    fn test_merge_counts_each_sample_once() {
        // The midpoint of edge `e` of child `c`, in the parent's [0, 1] frame
        // (an edge along axis t is indexed by its offsets along t + 1, t + 2)
        let mid = |c: usize, e: usize| -> [f32; 3] {
            let axis = e / 4;
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut p = [0.0f32; 3];
            for (k, p) in p.iter_mut().enumerate() {
                *p = ((c >> k) & 1) as f32 * 0.5;
            }
            p[axis] += 0.25;
            p[u] += (e & 1) as f32 * 0.5;
            p[v] += ((e >> 1) & 1) as f32 * 0.5;
            p
        };
        let on_parent_edge = |p: [f32; 3]| {
            p.iter().filter(|&&x| x == 0.0 || x == 1.0).count() >= 2
        };
        let weight = |m: &LeafHermiteData| {
            m.center_qef.mass_point().w
                + m.face_qefs.iter().map(|f| f.mass_point().w).sum::<f64>()
        };
        let mut segments: Vec<[f32; 3]> = vec![];
        for c in 0..8 {
            for e in 0..12 {
                let p = mid(c, e);
                if !on_parent_edge(p) && !segments.contains(&p) {
                    segments.push(p);
                }
            }
        }
        // 6 faces x 4 half-edges + 6 half-edges through the centre
        assert_eq!(segments.len(), 30);
        for s in &segments {
            let mut leafs = [LeafHermiteData::default(); 8];
            for (c, leaf) in leafs.iter_mut().enumerate() {
                for e in 0..12 {
                    if mid(c, e) == *s {
                        leaf.intersections[e] = LeafIntersection {
                            pos: nalgebra::Vector4::new(s[0], s[1], s[2], 1.0),
                            grad: nalgebra::Vector4::new(1.0, 0.0, 0.0, 0.0),
                        };
                    }
                }
            }
            let m = LeafHermiteData::merge(leafs).unwrap();
            assert_eq!(weight(&m), 1.0, "inner edge segment at {s:?}");
        }
        // A child's face QEF (face f = 2 t + side) on the parent's surface
        // counts once; an inner face is shared by two children (their QEFs
        // hold the same samples), so the pair counts once
        let face_weight = |c: usize, f: usize| {
            let mut leafs = [LeafHermiteData::default(); 8];
            leafs[c].face_qefs[f].add_intersection(
                nalgebra::Vector3::zeros(),
                nalgebra::Vector4::new(1.0, 0.0, 0.0, 0.0),
            );
            weight(&LeafHermiteData::merge(leafs).unwrap())
        };
        for c in 0..8 {
            for f in 0..6 {
                let (t, side) = (f / 2, f % 2);
                if (c >> t) & 1 == side {
                    assert_eq!(face_weight(c, f), 1.0, "child {c}, face {f}");
                } else if side == 1 {
                    let twin = face_weight(c ^ (1 << t), f ^ 1);
                    assert_eq!(
                        face_weight(c, f) + twin,
                        1.0,
                        "child {c}, face {f}"
                    );
                }
            }
        }
    }

    /// Mean and largest |field| at the triangles' centres, over the cell size
    fn centroid_error(
        shape: &BoundShape<VmFunction, f32>,
        mesh: &Mesh,
    ) -> (f32, f32) {
        let tape = shape.shape().ez_point_tape();
        let mut eval = VmShape::new_point_eval();
        let (mut sum, mut worst) = (0.0f32, 0.0f32);
        for t in &mesh.triangles {
            let c =
                (mesh.vertices[t.x] + mesh.vertices[t.y] + mesh.vertices[t.z])
                    / 3.0;
            let v = eval.eval(&tape, c.x, c.y, c.z).unwrap().0.abs();
            sum += v;
            worst = worst.max(v);
        }
        (sum / mesh.triangles.len() as f32, worst)
    }

    /// `max_err` is a distance in model units: an octree twice as large (one
    /// level deeper, same cell size) collapses the same surface to about the
    /// same mesh, and the mesh stays within the tolerance
    #[test]
    fn test_max_err_is_absolute() {
        let shape = tree_to_shape(tilted_torus());
        let tol = 0.01;
        let mesh_at = |half: f32, depth: u8, max_err| {
            let settings = Settings {
                depth,
                threads: None,
                world_to_model: nalgebra::Matrix4::new_scaling(half),
                max_err,
                ..Default::default()
            };
            Octree::build(&shape, &settings).unwrap().walk_dual()
        };
        let small = mesh_at(1.0, 6, Some(tol));
        let large = mesh_at(2.0, 7, Some(tol));
        let (a, b) =
            (small.triangles.len() as f32, large.triangles.len() as f32);
        assert!((a / b - 1.0).abs() < 0.1, "{a} vs {b} triangles");
        for m in [&small, &large] {
            check_for_vertex_dupes(m).unwrap();
            check_for_edge_matching(m).unwrap();
            let (mean, worst) = centroid_error(&shape, m);
            assert!(mean < tol && worst < 4.0 * tol, "{mean} {worst}");
        }
        // A tighter tolerance keeps more triangles; none at all collapses no
        // surface cell
        let tight = mesh_at(1.0, 6, Some(tol / 10.0));
        let none = mesh_at(1.0, 6, Some(0.0));
        assert!(tight.triangles.len() as f32 > 1.5 * a);
        assert!(none.triangles.len() >= tight.triangles.len());
    }
}
