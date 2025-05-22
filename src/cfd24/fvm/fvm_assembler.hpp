#ifndef CFD_FVM_ASSEMBLER_HPP
#define CFD_FVM_ASSEMBLER_HPP

#include "cfd24/mat/lodmat.hpp"
#include "cfd24/grid/i_grid.hpp"

namespace cfd{

///////////////////////////////////////////////////////////////////////////////
// FvmExtendedCollocations
///////////////////////////////////////////////////////////////////////////////

struct FvmExtendedCollocations{
public:
	FvmExtendedCollocations(const IGrid& grid, const std::vector<size_t>& ignored_bnd_faces={});

	std::vector<Point> points;
	std::vector<size_t> cell_collocations;
	std::vector<size_t> face_collocations;

	/// number of collocations
	size_t size() const;

	/// index of a cell for the given collocation. Throws if icolloc is not a cell collocation
	size_t cell_index(size_t icolloc) const;

	/// index of a face for the given collocation. Throws if icolloc is not a face collocation
	size_t face_index(size_t icolloc) const;
	
	std::array<size_t, 2> tab_face_colloc(size_t iface) const;

	std::vector<size_t> tab_colloc_colloc(size_t icolloc) const;

private:
	std::vector<std::array<size_t, 2>> _tab_face_colloc;
	std::vector<std::vector<size_t>> _tab_colloc_colloc;
	std::vector<size_t> _face_indices;
};

///////////////////////////////////////////////////////////////////////////////
// FvmCellGradient
///////////////////////////////////////////////////////////////////////////////

struct IFvmCellGradient{
	virtual ~IFvmCellGradient() = default;
	std::vector<Vector> compute(const std::vector<double>& u) const;
protected:
	IFvmCellGradient(std::array<CsrMatrix, 3>&& data): _data(std::move(data)) {};
private:
	std::array<CsrMatrix, 3> _data;
};

struct FvmCellGradient_LeastSquares: public IFvmCellGradient{
	FvmCellGradient_LeastSquares(const IGrid& grid, const FvmExtendedCollocations& colloc);

};

struct FvmCellGradient_Gauss: public IFvmCellGradient{
	FvmCellGradient_Gauss(const IGrid& grid, const FvmExtendedCollocations& colloc);
};

///////////////////////////////////////////////////////////////////////////////
// DfDn on faces
///////////////////////////////////////////////////////////////////////////////

/// DfDn on faces computer
struct FvmFacesDn{
	FvmFacesDn(const IGrid& grid, const FvmExtendedCollocations& colloc);

	/// computes dfdn for each grid face
	std::vector<double> compute(const std::vector<double>& f) const;

	/// computes dfdn for the given grid face
	double compute(size_t iface, const std::vector<double>& f) const;

	/// returns dfdn as a linear combination of collocation values
	const std::map<size_t, double>& linear_combination(size_t iface) const;
private:
	LodMatrix _dfdn;
};

}
#endif
