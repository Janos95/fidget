//! Mesh builder data structure and implementation
use super::{Mesh, Octree, cell::CellIndex, dc, frame::Frame};

/// Container used during construction of a [`Mesh`]
#[derive(Default)]
pub struct MeshBuilder {
    /// Map from indexes in [`Octree::verts`](super::Octree::verts) to
    /// `out.vertices`
    ///
    /// `usize::MAX` is used a marker for an unmapped vertex
    map: Vec<usize>,
    /// Vertices added on ambiguous faces ([`dc::dc_edge`]), by the positions
    /// of the inside corner that their arc cuts off and of the face's corner
    /// diagonal from it
    arcs: std::collections::HashMap<[u32; 6], usize>,
    out: Mesh,
}

impl MeshBuilder {
    pub fn take(self) -> Mesh {
        self.out
    }

    pub(crate) fn cell(&mut self, octree: &Octree, cell: CellIndex<3>) {
        dc::dc_cell(octree, cell, self);
    }

    pub(crate) fn face<F: Frame>(
        &mut self,
        octree: &Octree,
        a: CellIndex<3>,
        b: CellIndex<3>,
    ) {
        dc::dc_face::<F>(octree, a, b, self)
    }

    /// Handles four cells that share a common edge aligned on axis `T`
    ///
    /// Cells positions are in the order `[0, U, U | V, U]`, i.e. a right-handed
    /// winding about `+T` (where `T, U, V` is a right-handed coordinate frame)
    pub(crate) fn edge<F: Frame>(
        &mut self,
        octree: &Octree,
        a: CellIndex<3>,
        b: CellIndex<3>,
        c: CellIndex<3>,
        d: CellIndex<3>,
    ) {
        dc::dc_edge::<F>(octree, a, b, c, d, self)
    }

    /// Record the given triangle
    ///
    /// Vertices are indices given by calls to [`Self::vertex`]
    ///
    /// The vertices are given in a clockwise winding with the intersection
    /// vertex (i.e. the one on the edge) always last.
    pub(crate) fn triangle(&mut self, a: usize, b: usize, c: usize) {
        self.out.triangles.push(nalgebra::Vector3::new(a, b, c))
    }

    /// Looks up (or adds) the vertex for an arc of surface across a face
    pub(crate) fn arc_vertex(
        &mut self,
        key: [u32; 6],
        pos: impl FnOnce() -> (nalgebra::Vector3<f32>, nalgebra::Vector3<f32>),
    ) -> usize {
        *self.arcs.entry(key).or_insert_with(|| {
            let (p, n) = pos();
            self.out.vertices.push(p);
            self.out.normals.push(n);
            self.out.vertices.len() - 1
        })
    }

    /// Looks up the given vertex, localizing it within a cell
    ///
    /// `v` is an absolute offset into `verts`, which should be a reference to
    /// [`Octree::verts`](super::Octree::verts).
    pub(crate) fn vertex(&mut self, v: usize, octree: &Octree) -> usize {
        if v >= self.map.len() {
            self.map.resize(v + 1, usize::MAX);
        }
        match self.map[v] {
            usize::MAX => {
                let next_vert = self.out.vertices.len();
                self.out.vertices.push(octree.verts[v].pos);
                self.out.normals.push(octree.normals[v]);
                self.map[v] = next_vert;

                next_vert
            }
            u => u,
        }
    }
}
