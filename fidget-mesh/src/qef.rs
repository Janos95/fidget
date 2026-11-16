use super::cell::CellVertex;

/// Solver for a quadratic error function to position a vertex within a cell
///
/// Sums are kept in `f64`: the error is a small difference of large terms
/// (planes far from the origin), and an absolute collapse tolerance
/// ([`Settings::max_err`](crate::Settings::max_err)) compares it to values far
/// below what `f32` resolves there.
#[derive(Copy, Clone, Debug, Default)]
pub struct QuadraticErrorSolver {
    /// A^T A term
    ata: nalgebra::Matrix3<f64>,

    /// A^T B term
    atb: nalgebra::Vector3<f64>,

    /// B^T B term
    btb: f64,

    /// Mass point of intersections is stored as XYZ / W, so that summing works
    mass_point: nalgebra::Vector4<f64>,
}

impl std::ops::AddAssign for QuadraticErrorSolver {
    fn add_assign(&mut self, rhs: Self) {
        self.ata += rhs.ata;
        self.atb += rhs.atb;
        self.btb += rhs.btb;
        self.mass_point += rhs.mass_point;
    }
}

impl QuadraticErrorSolver {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn mass_point(&self) -> nalgebra::Vector4<f64> {
        self.mass_point
    }

    /// Adds a new intersection to the QEF
    ///
    /// `pos` is the position of the intersection and is accumulated in the mass
    /// point.  `grad` is the gradient at the surface, and is normalized in this
    /// function.
    pub fn add_intersection(
        &mut self,
        pos: nalgebra::Vector3<f32>,
        grad: nalgebra::Vector4<f32>,
    ) {
        // TODO: correct for non-zero distance value in grad.w
        let pos = pos.map(f64::from);
        self.mass_point += nalgebra::Vector4::new(pos.x, pos.y, pos.z, 1.0);
        // A zero gradient (e.g. where an exact distance field is exactly zero
        // and every term's derivative vanishes) gives no plane: normalizing it
        // would poison the whole QEF with NaN, so only the position counts.
        let Some(norm) =
            grad.xyz().map(f64::from).try_normalize(f64::MIN_POSITIVE)
        else {
            return;
        };
        self.ata += norm * norm.transpose();
        self.atb += norm * norm.dot(&pos);
        self.btb += norm.dot(&pos).powi(2);
    }

    /// Solve the given QEF, minimizing towards the mass point
    ///
    /// Returns a vertex localized within the given cell, and adjusts the solver
    /// to increase the likelihood that the vertex is bounded in the cell.
    ///
    /// Also returns the QEF error as the second item in the tuple, clamped to
    /// a small > 0 value (see [`solve_exact`](Self::solve_exact) for the
    /// error itself)
    pub fn solve(&self) -> (CellVertex<3>, f32) {
        let (pos, err) = self.solve_exact();
        (pos, (err as f32).max(1e-6))
    }

    /// Like [`solve`](Self::solve), with the error unclamped: the sum of the
    /// squared distances from the vertex to the intersections' planes
    pub fn solve_exact(&self) -> (CellVertex<3>, f64) {
        // This gets a little tricky; see
        // https://www.mattkeeter.com/projects/qef for a walkthrough of QEF math
        // and references to primary sources.
        let center = self.mass_point.xyz() / self.mass_point.w;
        let atb = self.atb - self.ata * center;

        let svd = nalgebra::linalg::SVD::new(self.ata, true, true);

        // nalgebra doesn't always actually order singular values (?!?)
        // https://github.com/dimforge/nalgebra/issues/1215
        let mut singular_values =
            svd.singular_values.data.0[0].map(ordered_float::OrderedFloat);
        singular_values.sort();
        singular_values.reverse();
        let singular_values = singular_values.map(|o| o.0);

        // Skip any eigenvalues that are small relative to the maximum
        // eigenvalue.  This is very much a tuned value (alas!).  If the value
        // is too small, then we incorrectly pick high-rank solutions, which may
        // shoot vertices out of their cells in near-planar situations.  If the
        // value is too large, then we incorrectly pick low-rank solutions,
        // which makes us less likely to snap to sharp features.
        //
        // For example, our cone test needs to use a rank-3 solver for
        // eigenvalues of [1.5633028, 1.430821, 0.0058764853] (a dynamic range
        // of 2e3); while the bear model needs to use a rank-2 solver for
        // eigenvalues of [2.87, 0.13, 5.64e-7] (a dynamic range of 10^7).  We
        // pick 10^3 here somewhat arbitrarily to be within that range.
        const EIGENVALUE_CUTOFF_RELATIVE: f64 = 1e-3;
        let cutoff = singular_values[0].abs() * EIGENVALUE_CUTOFF_RELATIVE;

        // Intuition about `rank`:
        // 0 => all eigenvalues are invalid (?!), use the center point
        // 1 => the first eigenvalue is valid, this must be planar
        // 2 => the first two eigenvalues are valid, this is a planar or an edge
        // 3 => all eigenvalues are valid, this is a planar, edge, or corner
        let rank = (0..3)
            .find(|i| singular_values[*i].abs() < cutoff)
            .unwrap_or(3);

        let epsilon = singular_values.get(rank).cloned().unwrap_or(0.0);
        let sol = svd.solve(&atb, epsilon);
        let mut pos = sol.map(|c| c + center).unwrap_or(center);
        if !pos.iter().all(|v| v.is_finite()) {
            pos = center;
        }
        // The error about the mass point, where its terms are small (about
        // the origin they are large and nearly cancel)
        let d = pos - center;
        let c0 = self.btb - 2.0 * center.dot(&self.atb)
            + (center.transpose() * self.ata * center)[0];
        let err = ((d.transpose() * self.ata * d)[0] - 2.0 * d.dot(&atb) + c0)
            .max(0.0);

        (
            CellVertex {
                pos: pos.map(|v| v as f32),
            },
            err,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::{Vector3, Vector4};

    #[test]
    fn qef_rank2() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.75, -0.75),
            Vector4::new(0.24, 0.12, 0.0, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.75, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.50, -1.0, -0.6),
            Vector4::new(0.0, 0.0, 0.31, 0.0),
        );
        let (_out, err) = q.solve();
        assert_eq!(err, 1e-6);
    }

    #[test]
    fn qef_zero_gradient() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(0.5, 0.0, 0.0),
            Vector4::new(1.0, 0.0, 0.0, 0.0),
        );
        q.add_intersection(
            Vector3::new(0.5, 0.5, 0.0),
            Vector4::new(0.0, 0.0, 0.0, 0.0),
        );
        let (out, _err) = q.solve();
        assert!(out.pos.iter().all(|v| v.is_finite()), "{:?}", out.pos);
        assert_eq!(out.pos.x, 0.5);
    }

    #[test]
    fn qef_near_planar() {
        let mut q = QuadraticErrorSolver::new();
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.4999981),
            Vector4::new(-0.66666776, -0.33333388, 0.66666526, -1.2516975e-6),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        q.add_intersection(
            Vector3::new(-0.5, -0.25, 0.50),
            Vector4::new(-0.6666667, -0.33333334, 0.6666667, 0.0),
        );
        let (out, err) = q.solve();
        assert_eq!(err, 1e-6);
        let expected = Vector3::new(-0.5, -0.25, 0.5);
        assert!(
            (out.pos - expected).norm() < 1e-3,
            "expected {expected:?}, got {:?}",
            out.pos
        );
    }
}
