//! Dual contouring implementation

use crate::{
    Octree,
    builder::MeshBuilder,
    cell::{Cell, CellIndex},
    frame::{Frame, XYZ, YZX, ZXY},
    types::{Corner, Edge, X, Y, Z},
};

pub fn dc_cell(octree: &Octree, cell: CellIndex<3>, out: &mut MeshBuilder) {
    if let Cell::Branch { .. } = octree[cell] {
        for i in Corner::iter() {
            out.cell(octree, octree.child(cell, i));
        }

        // Helper function for DC face calls
        fn dc_faces<T: Frame>(
            octree: &Octree,
            cell: CellIndex<3>,
            out: &mut MeshBuilder,
        ) {
            let (t, u, v) = T::frame();
            for c in [Corner::new(0), u.into(), v.into(), u | v] {
                out.face::<T>(
                    octree,
                    octree.child(cell, c),
                    octree.child(cell, c | t),
                );
            }
        }
        dc_faces::<XYZ>(octree, cell, out);
        dc_faces::<YZX>(octree, cell, out);
        dc_faces::<ZXY>(octree, cell, out);

        #[allow(unused_parens)]
        for i in [false, true] {
            out.edge::<XYZ>(
                octree,
                octree.child(cell, (X * i)),
                octree.child(cell, (X * i) | Y),
                octree.child(cell, (X * i) | Y | Z),
                octree.child(cell, (X * i) | Z),
            );
            out.edge::<YZX>(
                octree,
                octree.child(cell, (Y * i)),
                octree.child(cell, (Y * i) | Z),
                octree.child(cell, (Y * i) | X | Z),
                octree.child(cell, (Y * i) | X),
            );
            out.edge::<ZXY>(
                octree,
                octree.child(cell, (Z * i)),
                octree.child(cell, (Z * i) | X),
                octree.child(cell, (Z * i) | X | Y),
                octree.child(cell, (Z * i) | Y),
            );
        }
    }
}

/// Handles two cells which share a common face
///
/// `lo` is below `hi` on the `T` axis; the cells share a `UV` face where
/// `T-U-V` is a right-handed coordinate system.
pub fn dc_face<T: Frame>(
    octree: &Octree,
    lo: CellIndex<3>,
    hi: CellIndex<3>,
    out: &mut MeshBuilder,
) {
    if octree.is_leaf(lo) && octree.is_leaf(hi) {
        return;
    }
    let (t, u, v) = T::frame();
    out.face::<T>(
        octree,
        octree.child(lo, t),
        octree.child(hi, Corner::new(0)),
    );
    out.face::<T>(octree, octree.child(lo, t | u), octree.child(hi, u));
    out.face::<T>(octree, octree.child(lo, t | v), octree.child(hi, v));
    out.face::<T>(octree, octree.child(lo, t | u | v), octree.child(hi, u | v));
    #[allow(unused_parens)]
    for i in [false, true] {
        out.edge::<T::Next>(
            octree,
            octree.child(lo, (u * i) | t),
            octree.child(lo, (u * i) | v | t),
            octree.child(hi, (u * i) | v),
            octree.child(hi, (u * i)),
        );
        out.edge::<<T::Next as Frame>::Next>(
            octree,
            octree.child(lo, (v * i) | t),
            octree.child(hi, (v * i)),
            octree.child(hi, (v * i) | u),
            octree.child(lo, (v * i) | u | t),
        );
    }
}

/// Handles four cells that share a common edge aligned on axis `T`
///
/// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
/// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
///
/// - `dc_edge<X>` is `[0, Y, Y | Z, Z]`
/// - `dc_edge<Y>` is `[0, Z, Z | X, X]`
/// - `dc_edge<Z>` is `[0, X, X | Y, Y]`
pub fn dc_edge<T: Frame>(
    octree: &Octree,
    a: CellIndex<3>,
    b: CellIndex<3>,
    c: CellIndex<3>,
    d: CellIndex<3>,
    out: &mut MeshBuilder,
) {
    let cs = [a, b, c, d];
    if cs.iter().all(|v| octree.is_leaf(*v)) {
        // If any of the leafs are Empty or Full, then this edge can't
        // include a sign change.  TODO: can we make this any -> all if we
        // collapse empty / filled leafs into Empty / Full cells?
        let leafs = cs.map(|cell| match octree[cell] {
            Cell::Leaf(leaf) => Some(leaf),
            Cell::Empty | Cell::Full => None,
            Cell::Branch { .. } => unreachable!(),
            Cell::Invalid => panic!(),
        });
        if leafs.iter().any(Option::is_none) {
            return;
        }
        let leafs = leafs.map(Option::unwrap);

        // TODO: should we pick a canonically deepest leaf instead of the first
        // among the four that's at the deepest depth?
        let deepest = (0..4).max_by_key(|i| cs[*i].depth).unwrap();

        let (t, _u, _v) = T::frame();

        // Each leaf has an edge associated with it
        #[allow(clippy::identity_op)]
        let edges = [
            Edge::new((t.index() * 4 + 3) as u8),
            Edge::new((t.index() * 4 + 2) as u8),
            Edge::new((t.index() * 4 + 0) as u8),
            Edge::new((t.index() * 4 + 1) as u8),
        ];

        // Find the starting sign of the relevant edge, bailing out early if
        // there isn't a sign change here.  All of the deepest edges should show
        // the same sign change, so it doesn't matter which one we pick here.
        let starting_sign = {
            let (start, end) = edges[deepest].corners();
            let start = !(leafs[deepest].mask & start);
            let end = !(leafs[deepest].mask & end);
            // If there is no sign change, then there's nothing to do here.
            if start == end {
                return;
            }
            start
        };

        // Iterate over each of the edges, assigning a vertex if the sign change
        // lines up.
        let mut verts = [None; 4];
        for i in 0..4 {
            if cs[i].depth == cs[deepest].depth {
                let (start, end) = edges[i].corners();
                let s = !(leafs[i].mask & start);
                let e = !(leafs[i].mask & end);
                debug_assert_eq!(s, starting_sign);
                debug_assert_eq!(e, !starting_sign);
                verts[i] = leafs[i].edge(edges[i]);
            } else {
                // We declare that only *manifold* leaf cells can be neighbors
                // to smaller leaf cells.  This means that there's only one
                // vertex to pick here.
                let mut iter =
                    (0..12).filter_map(|j| leafs[i].edge(Edge::new(j)));
                verts[i] = iter.next();
                if iter.any(|other| other.vert != verts[i].unwrap().vert) {
                    panic!("invalid leaf vertex at {a:?}");
                }
            }
        }

        let verts = verts.map(Option::unwrap);

        // The intersection vertex comes from the deepest cell; the cell
        // vertices go around the edge in the winding that makes triangles
        // face out, each once (two neighbours may be the same larger cell).
        //
        // As always, we have to sample the deepest leaf's edge to be sure that
        // we get the correct value.
        let center = leafs[deepest].index + verts[deepest].edge.0 as usize;
        let winding = if starting_sign { 3 } else { 1 };
        let mut ring = arrayvec::ArrayVec::<usize, 4>::new();
        let mut arcs =
            arrayvec::ArrayVec::<(usize, ([u32; 6], usize, usize)), 4>::new();
        for k in 0..4 {
            let j = (k * winding) % 4;
            let n = (j + winding) % 4;
            if cs[j].index != cs[n].index {
                ring.push(leafs[j].index + verts[j].vert.0 as usize);
                if let Some(a) = arc(&cs, &leafs, &edges, deepest, j, n) {
                    arcs.push((ring.len(), a));
                }
            }
        }
        if arcs.is_empty() {
            for [a, b, c] in triangulate(octree, &ring, center) {
                let a = out.vertex(a, octree);
                let b = out.vertex(b, octree);
                let c = out.vertex(c, octree);
                out.triangle(a, b, c);
            }
        } else {
            // A fan through the arcs' vertices
            let mut full = arrayvec::ArrayVec::<usize, 8>::new();
            let mut arcs = arcs.into_iter().peekable();
            for (k, &r) in ring.iter().enumerate() {
                full.push(out.vertex(r, octree));
                if let Some(&(_, (key, p, q))) =
                    arcs.peek().filter(|&&(after, _)| after == k + 1)
                {
                    full.push(out.arc_vertex(key, || {
                        let n = octree.normals[p] + octree.normals[q];
                        (
                            (octree.verts[p].pos + octree.verts[q].pos) / 2.0,
                            n.try_normalize(1e-12).unwrap_or(n),
                        )
                    }));
                    arcs.next();
                }
            }
            let c = out.vertex(center, octree);
            for k in 0..full.len() {
                out.triangle(full[k], full[(k + 1) % full.len()], c);
            }
        }
    } else {
        let (t, u, v) = T::frame();

        #[allow(unused_parens)]
        for i in [false, true] {
            out.edge::<T>(
                octree,
                octree.child(a, (t * i) | u | v),
                octree.child(b, (t * i) | v),
                octree.child(c, (t * i)),
                octree.child(d, (t * i) | u),
            )
        }
    }
}

/// Finds whether the surface needs a vertex of its own on the face between
/// ring cells `j` and `n` (around the edge `edges[deepest]`)
///
/// On an ambiguous face (inside corners diagonal from each other), the surface
/// crosses in two arcs, each cutting off an inside corner.  If the cells on
/// both sides use one vertex for both arcs (their surface joins the two arcs
/// inside each cell: a tunnel through the face), those two vertices alone
/// can't make the loop around the tunnel, and the edge between them would be
/// shared by four triangles.  Then each arc gets a vertex on the face, between
/// its two intersections, so the fans of the arc's two edges go through it.
///
/// Returns the arc's key and the octree indices of its two intersections.
#[allow(clippy::type_complexity)]
fn arc(
    cs: &[CellIndex<3>; 4],
    leafs: &[crate::cell::Leaf<3>; 4],
    edges: &[Edge; 4],
    deepest: usize,
    j: usize,
    n: usize,
) -> Option<([u32; 6], usize, usize)> {
    let depth = cs[deepest].depth;
    let s = [j, n].into_iter().find(|&i| cs[i].depth == depth)?;
    // Cells 0, 1, 2, 3 sit at (0, 0), (U, 0), (U, V), (0, V) around the edge,
    // whose index in each cell is `4 * t + u_bit + 2 * v_bit`: across a face
    // normal to U the opposite edge differs in its V bit, and vice versa
    let flip = match (j.min(n), j.max(n)) {
        (0, 1) | (2, 3) => 2,
        _ => 1,
    };
    let opposite = |i: usize| Edge::new(edges[i].index() as u8 ^ flip);

    // The face must be ambiguous: the inside ends of the two edges at
    // opposite ends along the edge axis
    let inside_corner = |e: Edge| {
        let (a, b) = e.corners();
        if leafs[s].mask & a { a } else { b }
    };
    let (ci, oi) = (inside_corner(edges[s]), inside_corner(opposite(s)));
    if (ci.index() ^ oi.index()).count_ones() != 2 {
        return None;
    }
    // Both cells use one vertex for both arcs (a larger cell has one vertex)
    let double = |i: usize| {
        cs[i].depth < depth
            || leafs[i].edge(edges[i]).map(|x| x.vert)
                == leafs[i].edge(opposite(i)).map(|x| x.vert)
    };
    if !double(j) || !double(n) {
        return None;
    }

    // The arc around the inside corner `ci`: from this edge to the edge from
    // `ci` across the face
    let (a, b) = edges[s].corners();
    let t = a.index() ^ b.index();
    let w = (ci.index() ^ oi.index()) & !t;
    let other = (0..12).map(Edge::new).find(|e| {
        let (x, y) = e.corners();
        [x.index(), y.index()] == [ci.index(), ci.index() ^ w]
            || [y.index(), x.index()] == [ci.index(), ci.index() ^ w]
    })?;
    let p = leafs[s].index + leafs[s].edge(edges[s])?.edge.0 as usize;
    let q = leafs[s].index + leafs[s].edge(other)?.edge.0 as usize;
    let c0 = cs[s].corner(ci);
    let c1 = cs[s].corner(Corner::new((ci.index() ^ t ^ w) as u8));
    let key = [
        c0[0].to_bits(),
        c0[1].to_bits(),
        c0[2].to_bits(),
        c1[0].to_bits(),
        c1[1].to_bits(),
        c1[2].to_bits(),
    ];
    Some((key, p, q))
}

/// Triangulates the polygon around one edge (cell vertices `ring`, in order,
/// and the edge's intersection vertex `center`; indices into the octree's
/// vertices)
///
/// The ring is split by one of its diagonals (two triangles for a quad, or the
/// ring itself when it is a triangle): the split that folds the least, then the
/// flatter of the two.  A fan around the intersection vertex is used only when
/// it scores better than that split.  The fan has twice the triangles, and its
/// centre lies on the surface while the cell vertices sit slightly off it (the
/// QEF minimum), so fans zig-zag: slivers and facet noise in shading.  The fan
/// still wins where both splits fold: where the octree's cells differ in size
/// a larger cell's vertex can sit so that a split faces against the surface.
/// The edges shared with the neighbouring polygons don't change, so the mesh
/// stays watertight.
fn triangulate(
    octree: &Octree,
    ring: &[usize],
    center: usize,
) -> arrayvec::ArrayVec<[usize; 3], 4> {
    let n = ring.len();
    let fan: arrayvec::ArrayVec<[usize; 3], 4> = (0..n)
        .map(|j| [ring[j], ring[(j + 1) % n], center])
        .collect();
    if n < 3 {
        return fan;
    }
    // Score: triangles with no area (a vertex on another, or three in a
    // line: they happen where the surface runs exactly through a cell's edge
    // or corner), then area that faces against the field's normals at the
    // corners. The fan is kept when it scores (0, 0).
    let score = |tris: &[[usize; 3]]| -> (usize, f32) {
        tris.iter()
            .map(|&[a, b, c]| {
                let (pa, pb, pc) = (
                    octree.verts[a].pos,
                    octree.verts[b].pos,
                    octree.verts[c].pos,
                );
                let x = (pb - pa).cross(&(pc - pa));
                let longest = (pb - pa)
                    .norm_squared()
                    .max((pc - pb).norm_squared())
                    .max((pa - pc).norm_squared());
                let flat = x.norm() <= 1e-4 * longest;
                let r =
                    octree.normals[a] + octree.normals[b] + octree.normals[c];
                let r = r.try_normalize(1e-6).unwrap_or_default();
                (flat as usize, (-x.dot(&r)).max(0.0))
            })
            .fold((0, 0.0), |(n, f), (m, g)| (n + m, f + g))
    };
    let fan_score = score(&fan);
    let splits: arrayvec::ArrayVec<arrayvec::ArrayVec<[usize; 3], 4>, 2> = if n
        == 3
    {
        [[[ring[0], ring[1], ring[2]]].into_iter().collect()]
            .into_iter()
            .collect()
    } else {
        (0..2)
            .map(|k| {
                let q = [ring[k], ring[k + 1], ring[k + 2], ring[(k + 3) % 4]];
                [[q[0], q[1], q[2]], [q[2], q[3], q[0]]]
                    .into_iter()
                    .collect()
            })
            .collect()
    };
    // How flat a split is: the cosine between its two triangles' normals
    let flatness = |tris: &[[usize; 3]]| -> f32 {
        let ns: arrayvec::ArrayVec<nalgebra::Vector3<f32>, 4> = tris
            .iter()
            .map(|&[a, b, c]| {
                let (pa, pb, pc) = (
                    octree.verts[a].pos,
                    octree.verts[b].pos,
                    octree.verts[c].pos,
                );
                (pb - pa)
                    .cross(&(pc - pa))
                    .try_normalize(1e-12)
                    .unwrap_or_default()
            })
            .collect();
        if ns.len() == 2 {
            ns[0].dot(&ns[1])
        } else {
            1.0
        }
    };
    const EPS: f32 = 1e-12;
    let mut pick: Option<(
        (usize, f32),
        f32,
        arrayvec::ArrayVec<[usize; 3], 4>,
    )> = None;
    for tris in splits {
        let (sc, fl) = (score(&tris), flatness(&tris));
        let better = match &pick {
            None => true,
            Some((bs, bf, _)) => {
                sc.0 < bs.0
                    || (sc.0 == bs.0
                        && (sc.1 < bs.1 - EPS
                            || ((sc.1 - bs.1).abs() <= EPS && fl > *bf)))
            }
        };
        if better {
            pick = Some((sc, fl, tris));
        }
    }
    match pick {
        Some((sc, _, tris))
            if sc.0 < fan_score.0
                || (sc.0 == fan_score.0 && sc.1 <= fan_score.1 + EPS) =>
        {
            tris
        }
        _ => fan,
    }
}
