#include "nan_signal_handler.hpp"
#include <errno.h>
#include <fenv.h>
#include <signal.h>
#include <stdexcept>

using namespace cfd::dbg;

#ifndef WIN32

namespace{

class Impl{
public:
	Impl();

	/// get singleton instance
	static Impl& instance();

	void start(){
		feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	}

	void stop(){
		fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	}

private:
	//exception which will be generated on floating point error
	struct E_SignalException;

	/**
	* Returns the bool flag indicating whether we received an exit signal
	* @return Flag indicating shutdown of program
	*/
	static bool got_exit_signal(){
		return instance()._got_exit_signal;
	}

	/**
	* Sets the bool flag indicating whether we received an exit signal
	*/
	static void set_exit_signal(bool exit_signal){
		instance()._got_exit_signal = exit_signal;
	}

	/**
	* Sets exit signal to true.
	* @param[in] ignored Not used but required by function prototype
	*                    to match required handler.
	*/
	static void exit_signal_handler(int ignored);

	//initialization
	void setup_signal_handlers();

	//true if signal was catched
	bool _got_exit_signal;
};

struct Impl::E_SignalException: public std::runtime_error{
	E_SignalException() noexcept: std::runtime_error("NanSignalHandler: Floating point error"){}
};

Impl::Impl(){
	try{
		_got_exit_signal = false;
		feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
		setup_signal_handlers();
	} catch (...){
		throw std::runtime_error("Error during NanSignalHandler initialisation");
	}
}

Impl& Impl::instance(){
	static Impl hand;
	return hand;
}

void Impl::exit_signal_handler(int ignored){
	instance()._got_exit_signal = true;
	throw E_SignalException();
}

void Impl::setup_signal_handlers(){
	if (signal((int)SIGFPE, Impl::exit_signal_handler) == SIG_ERR){
		throw;
	}
}

} // namespace

// ======================= NanSignalHandler

void NanSignalHandler::start_check(){
	Impl::instance().start();
}

void NanSignalHandler::stop_check(){
	Impl::instance().stop();
}

#else
// empty implementation for windows platform

void NanSignalHandler::start_check(){}

void NanSignalHandler::stop_check(){}

#endif
