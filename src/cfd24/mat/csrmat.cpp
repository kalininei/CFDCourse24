#include "csrmat.hpp"

using namespace cfd;

void CsrStencil::set_stencil(std::vector<size_t>&& addr, std::vector<size_t>&& cols){
	_addr = std::move(addr);
	_cols = std::move(cols);
}

void CsrStencil::set_stencil(const std::vector<std::set<size_t>>& stencil_set){
	_addr = std::vector<size_t>(1, 0);
	_cols.clear();

	for (size_t irow=0; irow<stencil_set.size(); ++irow){
		const std::set<size_t>& cols = stencil_set[irow];
		_addr.push_back(_addr.back() + cols.size());
		for (size_t col: cols){
			_cols.push_back(col);
		}
	}
}

size_t CsrStencil::n_nonzeros() const{
	return _cols.size();
}

size_t CsrStencil::n_rows() const{
	return _addr.size()-1;
}

const std::vector<size_t>& CsrStencil::addr() const{
	return _addr;
}

const std::vector<size_t>& CsrStencil::cols() const{
	return _cols;
}

void CsrStencil::validate() const{
	// sizes
	if (_addr.size() < 1){
		throw std::runtime_error("addr array should have more then zero entries");
	}
	if (_cols.size() != _addr.back()){
		throw std::runtime_error("cols array size should match last addr entry");
	}
	// non-decreasing
	for (size_t i=1; i<_addr.size(); ++i){
		if (_addr[i] < _addr[i-1]){
			throw std::runtime_error("addr array should be non-decreasing");
		}
	}
}

bool CsrStencil::is_in_stencil(size_t irow, size_t icol) const{
	size_t start = _addr.at(irow);
	size_t end = _addr.at(irow+1);
	for (size_t i=start; i<end; ++i){
		if (_cols.at(i) == icol){
			return true;
		}
	}
	return false;
}

double CsrStencil::value(size_t irow, size_t icol) const{
	throw std::runtime_error("CsrStencil has no values");
}

std::vector<double> CsrStencil::mult_vec(const std::vector<double>& u) const{
	throw std::runtime_error("CsrStencil has no values");
}

double CsrStencil::mult_vec(size_t irow, const std::vector<double>& u) const{
	throw std::runtime_error("CsrStencil has no values");
}

size_t CsrStencil::get_address(size_t irow, size_t icol) const{
	std::vector<size_t>::const_iterator it_start = _cols.begin() + _addr.at(irow);
	std::vector<size_t>::const_iterator it_end = _cols.begin() + _addr.at(irow+1);
	auto fnd = std::lower_bound(it_start, it_end, icol);
	if (fnd != it_end && *fnd == icol){
		size_t a = fnd - _cols.begin();
		return a;
	}
	return INVALID_INDEX;
}

void CsrMatrix::set_values(std::vector<double>&& vals){
	_vals = std::move(vals);
}

const std::vector<double>& CsrMatrix::vals() const{
	return _vals;
}

std::vector<double>& CsrMatrix::vals(){
	return _vals;
}

void CsrMatrix::validate() const{
	CsrStencil::validate();

	// values size
	if (_vals.size() != n_nonzeros()){
		throw std::runtime_error("values array should have same size as the columns arrays");
	}
}

double CsrMatrix::value(size_t irow, size_t icol) const{
	size_t a = get_address(irow, icol);
	if (a != INVALID_INDEX){
		return _vals[a];
	} else {
		return 0.0;
	}
}

std::vector<double> CsrMatrix::mult_vec(const std::vector<double>& u) const{
	const std::vector<size_t>& a = addr();
	const std::vector<size_t>& c = cols();
	const std::vector<double>& v = vals();

	std::vector<double> ret(n_rows(), 0);
	for (size_t irow=0; irow<n_rows(); ++irow){
		size_t start = a[irow];
		size_t end = a[irow+1];
		for (size_t i=start; i<end; ++i){
			ret[irow] += v[i] * u[c[i]];
		}
	}

	return ret;
}

double CsrMatrix::mult_vec(size_t irow, const std::vector<double>& u) const{
	const std::vector<size_t>& a = addr();
	const std::vector<size_t>& c = cols();
	const std::vector<double>& v = vals();

	double ret = 0;
	size_t start = a.at(irow);
	size_t end = a.at(irow+1);
	for (size_t i=start; i<end; ++i){
		if (i >= v.size() || i >= c.size() || c[i] >= u.size()){
			throw std::runtime_error("BAD");
		}
		ret += v[i] * u[c[i]];
	}
	return ret;
}

void CsrMatrix::set_unit_row(size_t irow){
	const std::vector<size_t>& a = addr();
	const std::vector<size_t>& c = cols();

	size_t start = a.at(irow);
	size_t end = a.at(irow+1);
	for (size_t i=start; i<end; ++i){
		_vals[i] = (c[i] == irow) ? 1.0 : 0.0;
	}
}
