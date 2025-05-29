/**
 * @file
 * @brief Provides functionality for catching nan operations on Unix systems.
 * 
 * see http://www.yolinux.com/TUTORIALS/C++Signals.html
 */
#ifndef CFD_NAN_HANDLER_HPP
#define CFD_NAN_HANDLER_HPP

namespace cfd{
namespace dbg{

class NanSignalHandler{
public:
	NanSignalHandler(const NanSignalHandler&) = delete;
	NanSignalHandler& operator=(const NanSignalHandler&) = delete;

	static void start_check();
	static void stop_check();
};

}
}

#endif
