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
        for k in 0..4 {
            let j = (k * winding) % 4;
            if cs[j].index != cs[(j + winding) % 4].index {
                ring.push(leafs[j].index + verts[j].vert.0 as usize);
            }
        }
        for [a, b, c] in triangulate(octree, &ring, center) {
            let a = out.vertex(a, octree);
            let b = out.vertex(b, octree);
            let c = out.vertex(c, octree);
            out.triangle(a, b, c);
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

/// Triangulates the polygon around one edge (cell vertices `ring`, in order,
/// and the edge's intersection vertex `center`; indices into the octree's
/// vertices)
///
/// The usual triangulation is a fan around the intersection vertex.  Where the
/// octree's cells differ in size, a larger cell's vertex can sit so that the
/// fan folds over: some of its triangles face against the surface.  Then the
/// ring is split by one of its diagonals instead (or is a single triangle),
/// whichever folds the least area; the edges shared with the neighbouring
/// polygons don't change, so the mesh stays watertight.
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
    // Area that faces against the field's normals at the corners (0 if none)
    let folded = |tris: &[[usize; 3]]| -> f32 {
        tris.iter()
            .map(|&[a, b, c]| {
                let (pa, pb, pc) = (
                    octree.verts[a].pos,
                    octree.verts[b].pos,
                    octree.verts[c].pos,
                );
                let r =
                    octree.normals[a] + octree.normals[b] + octree.normals[c];
                let r = r.try_normalize(1e-6).unwrap_or_default();
                (-(pb - pa).cross(&(pc - pa)).dot(&r)).max(0.0)
            })
            .sum()
    };
    let mut best = folded(&fan);
    if best == 0.0 {
        return fan;
    }
    let mut out = fan;
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
    for tris in splits {
        let f = folded(&tris);
        if f < best {
            best = f;
            out = tris;
        }
    }
    out
}
