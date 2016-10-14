#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "runtimestatistics.h"

#include <random>
#include <string>

class PasswordGenerator
{
public:
    virtual void nextPassword(const void *&pw, std::size_t &pwSize) = 0;
};

class DummyPasswordGenerator : public PasswordGenerator
{
private:
    std::mt19937 gen;
    std::string currentPw;

    static constexpr std::size_t PASSWORD_LENGTH = 64;

public:
    DummyPasswordGenerator()
        : gen(std::chrono::system_clock::now().time_since_epoch().count())
    {
    }

    void nextPassword(const void *&pw, std::size_t &pwSize) override
    {
        currentPw.resize(PASSWORD_LENGTH);
        for (std::size_t i = 0; i < PASSWORD_LENGTH; i++) {
            currentPw[i] = (unsigned char)gen();
        }
        pw = currentPw.data();
        pwSize = currentPw.size();
    }
};

#include "argon2-opencl/argon2-common.h"

class BenchmarkDirector;

class Argon2Runner
{
public:
    virtual nanosecs runBenchmark(const BenchmarkDirector &director,
                                  PasswordGenerator &pwGen) = 0;
};

class BenchmarkDirector
{
private:
    std::string progname;
    argon2::Type type;
    argon2::Version version;
    std::size_t t_cost, m_cost, lanes;
    std::size_t batchSize, samples;
    std::string outputMode, outputType;
    bool beVerbose;

public:
    const std::string &getProgname() const { return progname; }
    argon2::Type getType() const { return type; }
    argon2::Version getVersion() const { return version; }
    std::size_t getTimeCost() const { return t_cost; }
    std::size_t getMemoryCost() const { return m_cost; }
    std::size_t getLanes() const { return lanes; }
    std::size_t getBatchSize() const { return batchSize; }
    bool isVerbose() const { return beVerbose; }

    BenchmarkDirector(const std::string &progname,
                      argon2::Type type, argon2::Version version,
                      std::size_t t_cost, std::size_t m_cost, std::size_t lanes,
                      std::size_t batchSize, std::size_t samples,
                      const std::string &outputMode,
                      const std::string &outputType)
        : progname(progname), type(type), version(version),
          t_cost(t_cost), m_cost(m_cost), lanes(lanes),
          batchSize(batchSize), samples(samples),
          outputMode(outputMode), outputType(outputType),
          beVerbose(outputMode == "verbose")
    {
    }

    int runBenchmark(Argon2Runner &runner) const;
};

class BenchmarkExecutive
{
public:
    virtual int runBenchmark(const BenchmarkDirector &director) const = 0;
};

#include "argon2-opencl/processingunit.h"

class OpenCLExecutive : public BenchmarkExecutive
{
private:
    class Runner : public Argon2Runner
    {
    private:
        argon2::Argon2Params params;
        argon2::opencl::ProcessingUnit unit;

    public:
        Runner(const BenchmarkDirector &director,
               const argon2::opencl::Device &device,
               const argon2::opencl::ProgramContext &pc);

        nanosecs runBenchmark(const BenchmarkDirector &director,
                              PasswordGenerator &pwGen) override;
    };

    static constexpr std::size_t HASH_LENGTH = 32;

    std::size_t deviceIndex;
    bool listDevices;

public:
    OpenCLExecutive(std::size_t deviceIndex, bool listDevices)
        : deviceIndex(deviceIndex), listDevices(listDevices)
    {
    }

    int runBenchmark(const BenchmarkDirector &director) const override;
};

#endif // BENCHMARK_H

