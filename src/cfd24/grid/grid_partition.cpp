#include "grid_partition.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"

using namespace cfd;

namespace{

void expand_domain(const IGrid& grid, std::vector<size_t>& cells){
	std::vector<bool> used(grid.n_cells(), false);
	for (size_t i=0; i<cells.size(); ++i){
		used[cells[i]] = true;
	}

	size_t old_cell_size = cells.size();
	for (size_t i=0; i<old_cell_size; ++i){
		size_t icell = cells[i];
		for (size_t ipoint: grid.tab_cell_point(icell)){
			for (size_t adjcell: grid.tab_point_cell(ipoint)){
				if (adjcell != INVALID_INDEX && used[adjcell] == false){
					used[adjcell] = true;
					cells.push_back(adjcell);
				}
			}
		}
	}
}

}

// =========================== SubGrid ========================================
namespace{
std::shared_ptr<IGrid> cut_grid2d(std::shared_ptr<IGrid> basic, const std::vector<size_t>& icells){
	std::vector<Point> points;
	std::vector<size_t> used_points(basic->n_points(), INVALID_INDEX);
	std::vector<std::vector<size_t>> tab_cell_point;
	for (size_t icell: icells){
		tab_cell_point.emplace_back();
		std::vector<size_t>& tab = tab_cell_point.back();
		for (size_t ipoint: basic->tab_cell_point(icell)){
			if (used_points[ipoint] == INVALID_INDEX){
				used_points[ipoint] = points.size();
				points.push_back(basic->point(ipoint));
			}
			tab.push_back(used_points[ipoint]);
		}
	}
	return std::make_shared<UnstructuredGrid2D>(points, tab_cell_point);
}

}
SubGrid::SubGrid(std::shared_ptr<IGrid> basic_grid, const std::vector<size_t>& icells, const std::vector<int>& buffer_layer_indexes, int ghost_depth)
		: _basic_grid(basic_grid), _buffer_layer_indexes(buffer_layer_indexes){

	std::vector<size_t> icells2 = icells;
	for (int ibuf=0; ibuf < ghost_depth; ++ibuf){
		expand_domain(*basic_grid, icells2);
	}
	for (size_t i=icells.size(); i < icells2.size(); ++i){
		_ghost_cells.push_back(i);
	}
	_global_cell_indexes = icells2;

	for (size_t i=icells.size(); i<icells2.size(); ++i){
		_buffer_layer_indexes.push_back(-1);
	}

	switch (dim()){
		case 2: _grid = cut_grid2d(basic_grid, icells2); break;
		default: _THROW_NOT_IMP_;
	}
}

size_t SubGrid::dim() const {
	return _basic_grid->dim();
}

size_t SubGrid::n_points() const {
	return _grid->n_points();
}

size_t SubGrid::n_cells() const {
	return _grid->n_cells();
}

size_t SubGrid::n_faces() const {
	return _grid->n_faces();
}

Point SubGrid::point(size_t ipoint) const {
	return _grid->point(ipoint);
}

Point SubGrid::cell_center(size_t icell) const {
	return _grid->cell_center(icell);
}

double SubGrid::cell_volume(size_t icell) const {
	return _grid->cell_volume(icell);
}

Vector SubGrid::face_normal(size_t iface) const {
	return _grid->face_normal(iface);
}

double SubGrid::face_area(size_t iface) const {
	return _grid->face_area(iface);
}

Point SubGrid::face_center(size_t iface) const {
	return _grid->face_center(iface);
}

std::vector<Point> SubGrid::points() const {
	return _grid->points();
}

std::vector<size_t> SubGrid::tab_cell_point(size_t icell) const {
	return _grid->tab_cell_point(icell);
}

std::array<size_t, 2> SubGrid::tab_face_cell(size_t iface) const {
	return _grid->tab_face_cell(iface);
}

std::vector<size_t> SubGrid::tab_face_point(size_t iface) const {
	return _grid->tab_face_point(iface);
}

std::vector<size_t> SubGrid::tab_cell_face(size_t icell) const {
	return _grid->tab_cell_face(icell);
}

void SubGrid::save_vtk(std::string fname) const {
	_grid->save_vtk(fname);
}

const std::vector<size_t>& SubGrid::ghost_cells() const{
	return _ghost_cells;
}

const std::vector<size_t>& SubGrid::domain_connection_faces() const{
	_cache.need_domain_connection_faces(*this);
	return _cache.domain_connection_faces;
}

size_t SubGrid::to_global_cell(size_t icell) const{
	return _global_cell_indexes[icell];
}

size_t SubGrid::to_global_face(size_t iface) const{
	_cache.need_global_face(*this);
	return _cache.global_face[iface];
}

std::shared_ptr<IGrid> SubGrid::basic_grid() const{
	return _basic_grid;
}

std::vector<double> SubGrid::expand_cell_vector(const std::vector<double>& u) const{
	std::vector<double> ret(_basic_grid->n_cells(), 0);
	for (size_t icell=0; icell < n_cells(); ++icell){
		size_t iglobal_cell = to_global_cell(icell);
		ret[iglobal_cell] = u[icell];
	}
	for (size_t ighost: ghost_cells()){
		size_t iglobal_ghost = to_global_cell(ighost);
		ret[iglobal_ghost] = 0.0;
	}
	return ret;
}

// =========================== SubGrid::Cache ================================
void SubGrid::Cache::clear(){
	domain_connection_faces.clear();
	global_face.clear();
}

void SubGrid::Cache::need_domain_connection_faces(const SubGrid& sg){
	if (!domain_connection_faces.empty()){
		return;
	}

	for (size_t iface: sg._grid->boundary_faces()){
		size_t iglobal_face = sg.to_global_face(iface);
		auto cc = sg._basic_grid->tab_face_cell(iglobal_face);
		if (cc[0] != INVALID_INDEX && cc[1] != INVALID_INDEX){
			domain_connection_faces.push_back(iface);
		}
	}
}

void SubGrid::Cache::need_global_face(const SubGrid& sg){
	if (!global_face.empty()){
		return;
	}

	auto g1 = sg._grid;
	auto g2 = sg._basic_grid;

	for (size_t iface1=0; iface1 < g1->n_faces(); ++iface1){
		Point center1 = g1->face_center(iface1);
		auto cc1 = g1->tab_face_cell(iface1);
		if (cc1[0] == INVALID_INDEX) {
			std::swap(cc1[0], cc1[1]);
		}
		auto cc2_0 = sg.to_global_cell(cc1[0]);
		for (size_t iface2: g2->tab_cell_face(cc2_0)){
			Point center2 = g2->face_center(iface2);
			if (vector_abs(center1 - center2) < 1e-6){
				global_face.push_back(iface2);
				break;
			}
		}
	}
}

// =========================== GridPartition ==================================
GridPartition::GridPartition(std::shared_ptr<IGrid> basic_grid, const std::vector<int>& location, int buffer_depth, int ghost_depth)
		: _grid(basic_grid){

	int ngrids = *std::max_element(location.begin(), location.end()) + 1;
	
	for (int igrid=0; igrid < ngrids; ++igrid){
		std::vector<size_t> cells;
		for (size_t icell=0; icell<basic_grid->n_cells(); ++icell){
			if (location[icell] == igrid){
				cells.push_back(icell);
			}
		}
		std::vector<int> buffer_layer_indexes(cells.size(), 0);
		for (int ibuf=0; ibuf < buffer_depth; ++ibuf){
			expand_domain(*basic_grid, cells);
			while (buffer_layer_indexes.size() < cells.size()){
				buffer_layer_indexes.push_back(ibuf + 1);
			}
		}

		_subgrids.push_back(std::make_shared<SubGrid>(basic_grid, cells, buffer_layer_indexes, ghost_depth));
	}
};

std::shared_ptr<SubGrid> GridPartition::subgrid(size_t igrid) const{
	return _subgrids.at(igrid);
}
size_t GridPartition::n_domains() const{
	return _subgrids.size();
}

GridPartition  GridPartition::build(
		std::shared_ptr<IGrid> basic_grid,
		std::function<int(Point)> location,
		int buffer_depth, int ghost_depth){

	std::vector<int> location_vec(basic_grid->n_cells());

	for (size_t icell=0; icell < basic_grid->n_cells(); ++icell){
		location_vec[icell] = location(basic_grid->cell_center(icell));
	}

	return GridPartition(basic_grid, location_vec, buffer_depth, ghost_depth);
}

template<int NDim>
GridPartition  GridPartition::build_uniform(
		std::shared_ptr<IGrid> basic_grid,
		std::array<int, NDim> npart,
		int buffer_depth, int ghost_depth){

	size_t nx=1, ny=1, nz=1;
	if constexpr (NDim > 0) nx = npart[0];
	if constexpr (NDim > 1) ny = npart[1];
	if constexpr (NDim > 2) nz = npart[2];

	std::vector<Point> points = basic_grid->points();
	double x0 = std::min_element(points.begin(), points.end(), [](Point a, Point b){ return a.x() < b.x(); })->x();
	double y0 = std::min_element(points.begin(), points.end(), [](Point a, Point b){ return a.y() < b.y(); })->y();
	double z0 = std::min_element(points.begin(), points.end(), [](Point a, Point b){ return a.z() < b.z(); })->z();
	double x1 = std::max_element(points.begin(), points.end(), [](Point a, Point b){ return a.x() < b.x(); })->x();
	double y1 = std::max_element(points.begin(), points.end(), [](Point a, Point b){ return a.y() < b.y(); })->y();
	double z1 = std::max_element(points.begin(), points.end(), [](Point a, Point b){ return a.z() < b.z(); })->z();

	double hx = (x1 - x0) / nx;
	double hy = (y1 - y0) / ny;
	double hz = (z1 - z0) / nz;

	auto location_func = [x0, y0, z0, nx, ny, nz, hx, hy, hz](Point p) -> int{
		size_t ix = std::min(size_t((p.x() - x0)/hx), nx-1);
		size_t iy = std::min(size_t((p.y() - y0)/hy), ny-1);
		size_t iz = std::min(size_t((p.z() - z0)/hz), nz-1);

		return ix + iy * nx + iz * nx * ny;
	};

	return build(basic_grid, location_func, buffer_depth, ghost_depth);
}

std::vector<double> GridPartition::restriction_weights(size_t idom, Restriction method) const{
	auto sgrid = _subgrids[idom];

	auto clear_ghost = [&](std::vector<double>& ret){
		for (size_t ighost: sgrid->ghost_cells()){
			ret[ighost] = 0.0;
		}
		return ret;
	};
	
	auto build_unity = [&](){
		std::vector<double> ret(sgrid->n_cells(), 1.0);
		return clear_ghost(ret);
	};

	auto build_original = [&](){
		std::vector<double> ret(sgrid->n_cells());
		for (size_t i=0; i<sgrid->n_cells(); ++i){
			ret[i] = (sgrid->buffer_layer_index(i) == 0) ? 1.0 : 0.0;
		}
		return ret;
	};

	auto build_average = [&](){
		std::vector<double> total(_grid->n_cells(), 0.0);
		for (size_t idom2=0; idom2 < n_domains(); ++idom2){
			for (size_t icell2=0; icell2 < _subgrids[idom2]->n_cells(); ++icell2){
				size_t iglobal_cell2 = _subgrids[idom2]->to_global_cell(icell2);
				total[iglobal_cell2] += 1.0;
			}
			for (size_t ighost: _subgrids[idom2]->ghost_cells()){
				size_t iglobal_ghost = _subgrids[idom2]->to_global_cell(ighost);
				total[iglobal_ghost] -= 1.0;
			}
		}

		std::vector<double> ret(sgrid->n_cells(), 0.0);
		for (size_t icell=0; icell<sgrid->n_cells(); ++icell){
			size_t iglobal_cell = sgrid->to_global_cell(icell);
			ret[icell] = 1.0 / total[iglobal_cell];
		}
		return clear_ghost(ret);
	};

	auto build_weighted_by_buffer_index = [&]() -> std::vector<double>{
		_THROW_NOT_IMP_;
	};

	switch (method){
		case Restriction::UNITY: return build_unity();
		case Restriction::ORIGINAL: return build_original();
		case Restriction::AVERAGE: return build_average();
		case Restriction::WEIGHTED_BY_BUFFER_LAYER_INDEX: return build_weighted_by_buffer_index();
		default: _THROW_NOT_IMP_;
	}
}




template GridPartition GridPartition::build_uniform<1>(std::shared_ptr<IGrid> basic_grid, std::array<int, 1> npart, int buffer_depth, int ghost_depth);
template GridPartition GridPartition::build_uniform<2>(std::shared_ptr<IGrid> basic_grid, std::array<int, 2> npart, int buffer_depth, int ghost_depth);
template GridPartition GridPartition::build_uniform<3>(std::shared_ptr<IGrid> basic_grid, std::array<int, 3> npart, int buffer_depth, int ghost_depth);
