#include "commandline/commandlineparser.h"
#include "commandline/argumenthandlers.h"

#include "benchmark.h"

#include <iostream>

using namespace libcommandline;

struct Arguments
{
    bool showHelp = false;
    bool listDevices = false;

    std::string mode = "opencl";

    std::size_t deviceIndex = 0;

    std::string outputType = "ns";
    std::string outputMode = "verbose";

    std::string type = "i";
    std::string version = "1.3";
    std::size_t t_cost = 1;
    std::size_t m_cost = 1024;
    std::size_t lanes = 1;
    std::size_t batchSize = 16;
    std::size_t sampleCount = 10;
};

static CommandLineParser<Arguments> buildCmdLineParser()
{
    static const auto positional = PositionalArgumentHandler<Arguments>(
                [] (Arguments &, const std::string &) {});

    std::vector<const CommandLineOption<Arguments>*> options {
        new FlagOption<Arguments>(
            [] (Arguments &state) { state.listDevices = true; },
            "list-devices", 'l', "list all available devices and exit"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.mode = mode; },
            "mode", 'm', "mode in which to run ('opencl' for OpenCL, 'cpu' for CPU)", "opencl", "MODE"),

        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t index) {
                state.deviceIndex = (std::size_t)index;
            }), "device", 'd', "use device with index INDEX", "0", "INDEX"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.outputType = type; },
            "output-type", 'o', "what to output (ns|ns-per-hash)", "ns", "TYPE"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &mode) { state.outputMode = mode; },
            "output-mode", '\0', "output mode (verbose|raw|mean|mean-and-mdev)", "verbose", "MODE"),

        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.type = type; },
            "type", 't', "Argon2 type (i|d)", "i", "TYPE"),
        new ArgumentOption<Arguments>(
            [] (Arguments &state, const std::string &type) { state.version = type; },
            "version", 'v', "Argon2 version (1.0|1.3)", "1.3", "VERSION"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.t_cost = (std::size_t)num;
            }), "t-cost", 'T', "time cost", "1", "N"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.m_cost = (std::size_t)num;
            }), "m-cost", 'M', "memory cost", "1024", "N"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.lanes = (std::size_t)num;
            }), "lanes", 'L', "number of lanes", "1", "N"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.batchSize = (std::size_t)num;
            }), "batch-size", 'b', "number of tasks per batch", "16", "N"),
        new ArgumentOption<Arguments>(
            makeNumericHandler<Arguments, std::size_t>([] (Arguments &state, std::size_t num) {
                state.sampleCount = (std::size_t)num;
            }), "samples", 's', "number of batches to run and measure", "10", "N"),

        new FlagOption<Arguments>(
            [] (Arguments &state) { state.showHelp = true; },
            "help", '?', "show this help and exit")
    };

    return CommandLineParser<Arguments>(
        "A tool for benchmarking the argon2-opencl library.",
        positional, options);
}

int main(int, const char * const *argv)
{
    CommandLineParser<Arguments> parser = buildCmdLineParser();

    Arguments args;
    int ret = parser.parseArguments(args, argv);
    if (ret != 0) {
        return ret;
    }
    if (args.showHelp) {
        parser.printHelp(argv);
        return 0;
    }

    argon2::Type type;
    if (args.type == "i") {
        type = argon2::ARGON2_I;
    } else if (args.type == "d") {
        type = argon2::ARGON2_D;
    } else {
        // TODO
        return 1;
    }

    argon2::Version version;
    if (args.version == "1.0") {
        version = argon2::ARGON2_VERSION_10;
    } else if (args.version == "1.3") {
        version = argon2::ARGON2_VERSION_13;
    } else {
        // TODO
        return 1;
    }

    BenchmarkDirector director(argv[0], type, version,
            args.t_cost, args.m_cost, args.lanes,
            args.batchSize, args.sampleCount,
            args.outputMode, args.outputType);
    if (args.mode == "opencl") {
        OpenCLExecutive exec(args.deviceIndex, args.listDevices);
        return exec.runBenchmark(director);
    } else if (args.mode == "cpu") {
        // TODO
        return 1;
    } else {
        std::cerr << argv[0] << ": invalid mode: " << args.mode << std::endl;
        return 1;
    }
}

