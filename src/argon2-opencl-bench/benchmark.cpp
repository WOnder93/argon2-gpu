#include "benchmark.h"

#include <iostream>

int BenchmarkDirector::runBenchmark(Argon2Runner &runner) const
{
    DummyPasswordGenerator pwGen;
    RunTimeStats stats(batchSize);
    for (std::size_t i = 0; i < samples; i++) {
        if (beVerbose) {
            std::cout << "  Sample " << i << "..." << std::endl;
        }
        stats.addSample(runner.runBenchmark(*this, pwGen));
    }
    stats.close();

    if (beVerbose) {
        auto &time = stats.getNanoseconds();
        std::cout << "Mean computation time: "
                  << RunTimeStats::repr((nanosecs)time.getMean())
                  << std::endl;
        std::cout << "Mean deviation: "
                  << RunTimeStats::repr((nanosecs)time.getMeanDeviation())
                  << " (" << time.getMeanDeviationPerMean() * 100.0 << "%)"
                  << std::endl;

        auto &perHash = stats.getNanosecsPerHash();
        std::cout << "Mean computation time (per hash): "
                  << RunTimeStats::repr((nanosecs)perHash.getMean())
                  << std::endl;
        std::cout << "Mean deviation (per hash): "
                  << RunTimeStats::repr((nanosecs)perHash.getMeanDeviation())
                  << std::endl;
        return 0;
    }

    const DataSet *dataSet;
    if (outputType == "ns") {
        dataSet = &stats.getNanoseconds();
    } else if (outputType == "ns-per-hash") {
        dataSet = &stats.getNanosecsPerHash();
    } else {
        std::cerr << progname << ": invalid output type: '"
                  << outputType << "'" << std::endl;
        return 1;
    }

    if (outputMode == "raw") {
        for (auto sample : dataSet->getSamples()) {
            std::cout << sample << std::endl;
        }
    } else if (outputMode == "mean") {
        std::cout << dataSet->getMean() << std::endl;
    } else if (outputMode == "mean-and-mdev") {
        std::cout << dataSet->getMean() << std::endl;
        std::cout << dataSet->getMeanDeviation() << std::endl;
    } else {
        std::cerr << progname << ": invalid output mode: '"
                  << outputMode << "'" << std::endl;
        return 1;
    }
    return 0;
}

OpenCLExecutive::Runner::Runner(
        const BenchmarkDirector &director,
        const argon2::opencl::Device &device,
        const argon2::opencl::ProgramContext &pc)
    : params(HASH_LENGTH, NULL, 0, NULL, 0, NULL, 0,
             director.getTimeCost(), director.getMemoryCost(),
             director.getLanes()),
      unit(&pc, &params, &device, director.getBatchSize())
{
}

nanosecs OpenCLExecutive::Runner::runBenchmark(
        const BenchmarkDirector &director, PasswordGenerator &pwGen)
{
    typedef std::chrono::steady_clock clock_type;
    using namespace argon2;
    using namespace argon2::opencl;

    auto beVerbose = director.isVerbose();
    auto batchSize = unit.getBatchSize();
    if (beVerbose) {
        std::cout << "Starting computation..." << std::endl;
    }

    clock_type::time_point checkpt0 = clock_type::now();
    {
        ProcessingUnit::PasswordWriter writer(unit);
        for (std::size_t i = 0; i < batchSize; i++) {
            const void *pw;
            std::size_t pwLength;
            pwGen.nextPassword(pw, pwLength);
            writer.setPassword(pw, pwLength);

            writer.moveForward(1);
        }
    }
    clock_type::time_point checkpt1 = clock_type::now();

    unit.beginProcessing();
    unit.endProcessing();

    clock_type::time_point checkpt2 = clock_type::now();
    {
        ProcessingUnit::HashReader reader(unit);
        for (std::size_t i = 0; i < batchSize; i++) {
            reader.getHash();
            reader.moveForward(1);
        }
    }
    clock_type::time_point checkpt3 = clock_type::now();

    if (beVerbose) {
        clock_type::duration wrTime = checkpt1 - checkpt0;
        auto wrTimeNs = toNanoseconds(wrTime);
        std::cout << "    Writing took     "
                  << RunTimeStats::repr(wrTimeNs) << std::endl;
    }

    clock_type::duration compTime = checkpt2 - checkpt1;
    auto compTimeNs = toNanoseconds(compTime);
    if (beVerbose) {
        std::cout << "    Computation took "
                  << RunTimeStats::repr(compTimeNs) << std::endl;
    }

    if (beVerbose) {
        clock_type::duration rdTime = checkpt3 - checkpt2;
        auto rdTimeNs = toNanoseconds(rdTime);
        std::cout << "    Reading took     "
                  << RunTimeStats::repr(rdTimeNs) << std::endl;
    }
    return compTimeNs;
}

int OpenCLExecutive::runBenchmark(const BenchmarkDirector &director) const
{
    using namespace argon2::opencl;

    GlobalContext global;
    auto &devices = global.getAllDevices();

    if (listDevices) {
        std::size_t i = 0;
        for (auto &device : devices) {
            std::cout << "Device #" << i << ": "
                      << device.getInfo() << std::endl;
            i++;
        }
        return 0;
    }
    if (deviceIndex > devices.size()) {
        std::cerr << director.getProgname()
                  << ": device index out of range: "
                  << deviceIndex << std::endl;
        return 1;
    }
    auto &device = devices[deviceIndex];
    if (director.isVerbose()) {
        std::cout << "Using device #" << deviceIndex << ": "
                  << device.getInfo() << std::endl;
    }
    ProgramContext pc(&global, { device },
                      director.getType(), director.getVersion());
    Runner runner(director, device, pc);
    return director.runBenchmark(runner);
}
