FDIR = functions
SDIR = solverloop
CC = icpx  # Use icpx for Intel's oneAPI DPC++ compiler
CFLAGS = -I. -fsycl

_FUNCTION_DEPS = global_vars.h functions.h matrix.h utility_functions.h \
                 filling.h reading_input_parameters.h free_variables.h fill_domain.h

DEPS = $(patsubst %,$(FDIR)/%,$(_FUNCTION_DEPS))

_SOLVERLOOP_DEPS = serialinfo_xy.h GibbsEnergyData.h functions_fftw.h file_writer.h

DEPS += $(patsubst %,$(SDIR)/%,$(_SOLVERLOOP_DEPS))

LIBS = -lfftw3 -lm

microsim_ch_fft: microsim_ch_fft.o
	$(CC) -o microsim_ch_fft microsim_ch_fft.o $(CFLAGS) $(LIBS)

microsim_ch_fft.o: $(DEPS)
	$(CC) -c microsim_ch_fft.cpp $(CFLAGS)

.PHONY: clean

clean:
	rm -f microsim_ch_fft.o microsim_ch_fft
