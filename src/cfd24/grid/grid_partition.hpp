#ifndef CFD_GRID_PARTITION_HPP
#define CFD_GRID_PARTITION_HPP

#include "cfd24/grid/i_grid.hpp"
#include <functional>

namespace cfd{

class GridPartition;

class SubGrid: public IGrid{
public:
	SubGrid(std::shared_ptr<IGrid> basic_grid, const std::vector<size_t>& icells, const std::vector<int>& buffer_layer_indexes, int ghost_depth);

	size_t dim() const override;
	size_t n_points() const override;
	size_t n_cells() const override;
	size_t n_faces() const override;
	Point point(size_t ipoint) const override;
	Point cell_center(size_t icell) const override;
	double cell_volume(size_t icell) const override;
	Vector face_normal(size_t iface) const override;
	double face_area(size_t iface) const override;
	Point face_center(size_t iface) const override;
	std::vector<Point> points() const override;
	std::vector<size_t> tab_cell_point(size_t icell) const override;
	std::array<size_t, 2> tab_face_cell(size_t iface) const override;
	std::vector<size_t> tab_face_point(size_t iface) const override;
	std::vector<size_t> tab_cell_face(size_t icell) const override;
	void save_vtk(std::string fname) const override;

	size_t to_global_cell(size_t icell) const;
	size_t to_global_face(size_t iface) const;
	const std::vector<size_t>& ghost_cells() const;
	const std::vector<size_t>& domain_connection_faces() const;
	int buffer_layer_index(size_t icell) const { return _buffer_layer_indexes[icell]; }
	std::vector<double> expand_cell_vector(const std::vector<double>& u) const;

	std::shared_ptr<IGrid> basic_grid() const;

private:
	std::shared_ptr<IGrid> _grid;
	std::shared_ptr<IGrid> _basic_grid;

	std::vector<size_t> _ghost_cells;
	std::vector<size_t> _global_cell_indexes;
	std::vector<int> _buffer_layer_indexes;

	struct Cache{
		std::vector<size_t> domain_connection_faces;
		std::vector<size_t> global_face;

		void clear();
		void need_domain_connection_faces(const SubGrid& sg);
		void need_global_face(const SubGrid& sg);
	};
	mutable Cache _cache;
};

class GridPartition{
public:
	GridPartition(std::shared_ptr<IGrid> basic_grid, const std::vector<int>& location, int buffer_depth, int ghost_depth);

	size_t n_domains() const;
	std::shared_ptr<SubGrid> subgrid(size_t igrid) const;

	enum struct Restriction{
		UNITY,
		ORIGINAL,
		AVERAGE,
		WEIGHTED_BY_BUFFER_LAYER_INDEX,
	};
	std::vector<double> restriction_weights(size_t idom, Restriction method) const;

	static GridPartition build(
			std::shared_ptr<IGrid> basic_grid,
			std::function<int(Point)> location,
			int buffer_depth, int ghost_depth);

	template<int NDim>
	static GridPartition build_uniform(
			std::shared_ptr<IGrid> basic_grid,
			std::array<int, NDim> npart,
			int buffer_depth, int ghost_depth);
private:
	std::shared_ptr<IGrid> _grid;
	std::vector<std::shared_ptr<SubGrid>> _subgrids;
};


}

#endif
