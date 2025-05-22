#include "fvm_assembler.hpp"
#include "cfd24/geom/searcher.hpp"
#include "cfd24/geom/simplex.hpp"
#include "cfd24/debug/tictoc.hpp"
#include "cfd24/mat/densemat.hpp"

using namespace cfd;

///////////////////////////////////////////////////////////////////////////////
// FvmExtendedCollocations
///////////////////////////////////////////////////////////////////////////////

FvmExtendedCollocations::FvmExtendedCollocations(const IGrid& grid, const std::vector<size_t>& ignored_bnd_faces){
	std::set<size_t> ignored_bnd_faces_set(ignored_bnd_faces.begin(), ignored_bnd_faces.end());

	// cell loop
	for (size_t icell=0; icell<grid.n_cells(); ++icell){
		// collocations at the cell center
		points.push_back(grid.cell_center(icell));
		cell_collocations.push_back(icell);
	}

	// face loop
	for (size_t iface=0; iface<grid.n_faces(); ++iface){
		std::array<size_t, 2> cells = grid.tab_face_cell(iface);

		// face -> collocation-points connectivity
		_tab_face_colloc.push_back({cells[0], cells[1]});

		// collocations at the boundary faces
		if (ignored_bnd_faces_set.find(iface) == ignored_bnd_faces_set.end())
		if (cells[0] == INVALID_INDEX || cells[1] == INVALID_INDEX){
			points.push_back(grid.face_center(iface));
			face_collocations.push_back(points.size()-1);
			_face_indices.push_back(iface);

			if (cells[0] == INVALID_INDEX){
				_tab_face_colloc.back()[0] = points.size()-1;
			} else {
				_tab_face_colloc.back()[1] = points.size()-1;
			}
		}
	}

	// collocations connectivity
	_tab_colloc_colloc.resize(points.size());
	for (const std::array<size_t, 2>& fc: _tab_face_colloc){
		if (fc[0] != INVALID_INDEX && fc[1] != INVALID_INDEX){
			_tab_colloc_colloc[fc[1]].push_back(fc[0]);
			_tab_colloc_colloc[fc[0]].push_back(fc[1]);
		}
	}
}

size_t FvmExtendedCollocations::size() const{
	return points.size();
}

size_t FvmExtendedCollocations::face_index(size_t icolloc) const{
	if (icolloc < cell_collocations.size()){
		_THROW_INTERNAL_ERROR_;
	} else {
		return _face_indices[icolloc - cell_collocations.size()];
	}
}

size_t FvmExtendedCollocations::cell_index(size_t icolloc) const{
	if (icolloc >= cell_collocations.size()){
		_THROW_INTERNAL_ERROR_;
	} else {
		return icolloc;
	}
}

std::array<size_t, 2> FvmExtendedCollocations::tab_face_colloc(size_t iface) const{
	return _tab_face_colloc[iface];
}

std::vector<size_t> FvmExtendedCollocations::tab_colloc_colloc(size_t icolloc) const{
	return _tab_colloc_colloc[icolloc];
}

///////////////////////////////////////////////////////////////////////////////
// Cell gradient Least squares
///////////////////////////////////////////////////////////////////////////////

std::vector<Vector> IFvmCellGradient::compute(const std::vector<double>& u) const{
	std::vector<double> x = _data[0].mult_vec(u);
	std::vector<double> y = _data[1].mult_vec(u);

	std::vector<Vector> ret(x.size());
	for (size_t i=0; i<ret.size(); ++i){
		ret[i].x() = x[i];
		ret[i].y() = y[i];
	}

	if (_data[2].n_rows() > 0){
		std::vector<double> z = _data[2].mult_vec(u);
		for (size_t i=0; i<ret.size(); ++i){
			ret[i].z() = z[i];
		}
	}

	return ret;
}


namespace{

DenseMatrix least_squares_inv(const DenseMatrix& a){
	// transpose(A)
	DenseMatrix at = a.transpose();
	// inverse(transpose(A) * A)
	DenseMatrix inv_at_a = at.mult_mat(a).inverse();
	// inverse(transpose(A) * A) * transpose(A)
	return inv_at_a.mult_mat(at);
}

std::array<CsrMatrix, 3> assemble_fvm_cell_gradient_2d(const IGrid& grid, const FvmExtendedCollocations& colloc){
	LodMatrix grad_x(grid.n_cells());
	LodMatrix grad_y(grid.n_cells());

	for (size_t icell = 0; icell < grid.n_cells(); ++icell){
		const std::vector<size_t>& collocs = colloc.tab_colloc_colloc(icell);

		DenseMatrix amat(collocs.size(), 2);
		for (size_t i=0; i<collocs.size(); ++i){
			Vector c = colloc.points[collocs[i]] - colloc.points[icell];
			amat.set_value(i, 0, c.x());
			amat.set_value(i, 1, c.y());
		}
		DenseMatrix lsi = least_squares_inv(amat);
		double diag_x = 0;
		double diag_y = 0;
		for (size_t i=0; i<collocs.size(); ++i){
			double vx = lsi.value(0, i);
			double vy = lsi.value(1, i);
			grad_x.set_value(icell, collocs[i], vx);
			grad_y.set_value(icell, collocs[i], vy);
			diag_x -= vx;
			diag_y -= vy;
		}
		grad_x.set_value(icell, icell, diag_x);
		grad_y.set_value(icell, icell, diag_y);
	}

	return {grad_x.to_csr(), grad_y.to_csr()};
}

std::array<CsrMatrix, 3> assemble_fvm_cell_gradient(const IGrid& grid, const FvmExtendedCollocations& colloc){
	if (grid.dim() == 2){
		return assemble_fvm_cell_gradient_2d(grid, colloc);
	} else {
		_THROW_NOT_IMP_;
	}
}

}

FvmCellGradient_LeastSquares::FvmCellGradient_LeastSquares(const IGrid& grid, const FvmExtendedCollocations& colloc)
	: IFvmCellGradient(assemble_fvm_cell_gradient(grid, colloc)){}

///////////////////////////////////////////////////////////////////////////////
// FvmCellGradient_Gauss
///////////////////////////////////////////////////////////////////////////////

namespace {

std::array<CsrMatrix, 3> assemble_fvm_cell_gradient_gauss(const IGrid& grid, const FvmExtendedCollocations& colloc){
	LodMatrix grad_x(grid.n_cells());
	LodMatrix grad_y(grid.n_cells());
	LodMatrix grad_z(grid.n_cells());

	for (size_t iface=0; iface < grid.n_faces(); ++iface){
		auto icollocs = colloc.tab_face_colloc(iface);
		Vector n = grid.face_area(iface) * grid.face_normal(iface);
		// ---- internal face
		if (icollocs[0] < grid.n_cells() && icollocs[1] < grid.n_cells()){
			const auto& icells = icollocs;
			double vol1 = grid.cell_volume(icells[0]);
			double vol2 = grid.cell_volume(icells[1]);
			// left cell
			grad_x.add_value(icells[0], icells[0], n.x()*0.5/vol1);
			grad_y.add_value(icells[0], icells[0], n.y()*0.5/vol1);
			grad_z.add_value(icells[0], icells[0], n.z()*0.5/vol1);
			grad_x.add_value(icells[0], icells[1], n.x()*0.5/vol1);
			grad_y.add_value(icells[0], icells[1], n.y()*0.5/vol1);
			grad_z.add_value(icells[0], icells[1], n.z()*0.5/vol1);
			// right cell
			grad_x.add_value(icells[1], icells[0], -n.x()*0.5/vol2);
			grad_y.add_value(icells[1], icells[0], -n.y()*0.5/vol2);
			grad_z.add_value(icells[1], icells[0], -n.z()*0.5/vol2);
			grad_x.add_value(icells[1], icells[1], -n.x()*0.5/vol2);
			grad_y.add_value(icells[1], icells[1], -n.y()*0.5/vol2);
			grad_z.add_value(icells[1], icells[1], -n.z()*0.5/vol2);
		}
		// boundary face (no right cell)
		else if (icollocs[1] >= grid.n_cells()){
			double vol1 = grid.cell_volume(icollocs[0]);
			grad_x.add_value(icollocs[0], icollocs[1], n.x()/vol1);
			grad_y.add_value(icollocs[0], icollocs[1], n.y()/vol1);
			grad_z.add_value(icollocs[0], icollocs[1], n.z()/vol1);
		// boundary face (no left cell)
		} else {
			double vol2 = grid.cell_volume(icollocs[1]);
			grad_x.add_value(icollocs[1], icollocs[0], -n.x()/vol2);
			grad_y.add_value(icollocs[1], icollocs[0], -n.y()/vol2);
			grad_z.add_value(icollocs[1], icollocs[0], -n.z()/vol2);
		}
	}
	std::array<CsrMatrix, 3> ret;
	ret[0] = grad_x.to_csr();
	ret[1] = grad_y.to_csr();
	if (grid.dim() > 2){
		ret[2] = grad_z.to_csr();
	}
	return ret;
}

};

FvmCellGradient_Gauss::FvmCellGradient_Gauss(const IGrid& grid, const FvmExtendedCollocations& colloc)
	: IFvmCellGradient(assemble_fvm_cell_gradient_gauss(grid, colloc)){

}

///////////////////////////////////////////////////////////////////////////////
// DuDn
///////////////////////////////////////////////////////////////////////////////

namespace {

LodMatrix assemble_faces_dudn_2d(const IGrid& grid, const FvmExtendedCollocations& colloc){
	LodMatrix mat(grid.n_faces());

	std::vector<std::vector<size_t>> tab_point_colloc(grid.n_points());
	{
		for (size_t icolloc: colloc.cell_collocations){
			size_t icell = colloc.cell_index(icolloc);
			for (size_t ipoint: grid.tab_cell_point(icell)){
				tab_point_colloc[ipoint].push_back(icolloc);
			}
		}
		for (size_t icolloc: colloc.face_collocations){
			size_t iface = colloc.face_index(icolloc);
			for (size_t ipoint: grid.tab_face_point(iface)){
				tab_point_colloc[ipoint].push_back(icolloc);
			}
		}
	}
	auto find_closest_collocation = [&](size_t grid_point, size_t excl0, size_t excl1){
		size_t ret = INVALID_INDEX;
		double min_meas = 1e100;
		Point p0 = grid.point(grid_point);
		for (size_t icolloc: tab_point_colloc[grid_point]){
			if (icolloc != excl0 && icolloc != excl1){
				double meas = vector_meas(p0 - colloc.points[icolloc]);
				if (meas < min_meas){
					min_meas = meas;
					ret = icolloc;
				}
			}
		}
		return ret;
	};

	auto add_ds_entry = [&](const Vector& normal, const Vector& c, size_t col0, size_t col1, size_t col2, size_t iface){
		Vector s(-normal.y(), normal.x());
		double cos_cos = dot_product(c, s) / dot_product(c, normal);

		Point p0 = colloc.points[col0];
		Point p1 = colloc.points[col1];
		Point p2 = colloc.points[col2];
		double tri_area = triangle_area(p0, p1, p2);

		double coef = -0.5*cos_cos/tri_area;

		double x0 = p0.x(); double y0 = p0.y();
		double x1 = p1.x(); double y1 = p1.y();
		double x2 = p2.x(); double y2 = p2.y();
		double dx0 = (y1 - y2)/2.0;
		double dy0 = (x2 - x1)/2.0;
		double dx1 = (y2 - y0)/2.0;
		double dy1 = (x0 - x2)/2.0;
		double dx2 = (y0 - y1)/2.0;
		double dy2 = (x1 - x0)/2.0;

		mat.add_value(iface, col0, coef*(dx0*s.x() + dy0*s.y()));
		mat.add_value(iface, col1, coef*(dx1*s.x() + dy1*s.y()));
		mat.add_value(iface, col2, coef*(dx2*s.x() + dy2*s.y()));
	};

	for (size_t iface = 0; iface < grid.n_faces(); ++iface){
		Vector normal = grid.face_normal(iface);
		size_t negative_collocation = colloc.tab_face_colloc(iface)[0];
		size_t positive_collocation = colloc.tab_face_colloc(iface)[1];
		Point ci = colloc.points[negative_collocation];
		Point cj = colloc.points[positive_collocation];
	
		// +dudc / cos(c,n);
		double v1 = 1.0/dot_product(normal, cj-ci);
		mat.set_value(iface, positive_collocation, v1);
		mat.set_value(iface, negative_collocation, -v1);

		// -duds*cos(c,s)/cos(c,n) 
		Vector c = (cj - ci)/vector_abs(cj - ci);
		{
			// left point
			size_t igrid = grid.tab_face_point(iface)[0];
			size_t col1 = find_closest_collocation(igrid, positive_collocation, negative_collocation);
			add_ds_entry(normal, c, negative_collocation, col1, positive_collocation, iface);
		}
		{
			// right point
			size_t igrid = grid.tab_face_point(iface)[1];
			size_t col1 = find_closest_collocation(igrid, positive_collocation, negative_collocation);
			add_ds_entry(normal, c, positive_collocation, col1, negative_collocation, iface);
		}
	}

	return mat;
}

LodMatrix build_dfdn_matrix(const IGrid& grid, const FvmExtendedCollocations& colloc){
	if (grid.dim() == 2){
		return assemble_faces_dudn_2d(grid, colloc);
	} else {
		_THROW_NOT_IMP_;
	}
}

}

FvmFacesDn::FvmFacesDn(const IGrid& grid, const FvmExtendedCollocations& colloc): _dfdn(build_dfdn_matrix(grid, colloc)){}

std::vector<double> FvmFacesDn::compute(const std::vector<double>& f) const{
	return _dfdn.mult_vec(f);
}

double FvmFacesDn::compute(size_t iface, const std::vector<double>& f) const{
	return _dfdn.mult_vec(iface, f);
}

const std::map<size_t, double>& FvmFacesDn::linear_combination(size_t iface) const{
	return _dfdn.row(iface);
}
