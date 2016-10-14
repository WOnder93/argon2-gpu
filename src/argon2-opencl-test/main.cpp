#include <iostream>
#include <cstdint>

#include "argon2-opencl/processingunit.h"

using namespace argon2;
using namespace argon2::opencl;

static char toHex(std::uint8_t digit) {
    return digit >= 10 ? 'a' + (digit - 10) : '0' + digit;
}

static void dumpBytes(std::ostream &out, const void *data, std::size_t size)
{
    auto bdata = static_cast<const std::uint8_t *>(data);
    for (std::size_t i = 0; i < size; i++) {
        auto val = bdata[i];
        out << toHex((val >> 4) & 0xf) << toHex(val & 0xf);
    }
}

class TestCase
{
private:
    Argon2Params params;
    const void *output;
    const void *input;
    std::size_t inputLength;

public:
    const Argon2Params &getParams() const { return params; }
    const void *getOutput() const { return output; }
    const void *getInput() const { return input; }
    std::size_t getInputLength() const { return inputLength; }

    TestCase(const Argon2Params &params, const void *output,
             const void *input, std::size_t inputLength)
        : params(params), output(output),
          input(input), inputLength(inputLength)
    {
    }

    void dump(std::ostream &out) const
    {
        out << "t=" << params.getTimeCost()
            << " m=" << params.getMemoryCost()
            << " p=" << params.getLanes()
            << " pass=";
        dumpBytes(out, input, inputLength);

        if (params.getSaltLength()) {
            out << " salt=";
            dumpBytes(out, params.getSalt(), params.getSaltLength());
        }

        if (params.getAssocDataLength()) {
            out << " ad=";
            dumpBytes(out, params.getAssocData(), params.getAssocDataLength());
        }

        if (params.getSecretLength()) {
            out << " secret=";
            dumpBytes(out, params.getSecret(), params.getSecretLength());
        }
    }
};

std::size_t runTests(const GlobalContext &global, const Device &device,
                     Type type, Version version,
                     const TestCase *casesFrom, const TestCase *casesTo)
{
    std::cerr << "Running tests for Argon2"
              << (type == ARGON2_I ? "i" : "d")
              << " v" << (version == ARGON2_VERSION_10 ? "1.0" : "1.3")
              << "..." << std::endl;

    std::size_t failures = 0;
    ProgramContext progCtx(&global, { device }, type, version);
    for (auto tc = casesFrom; tc < casesTo; ++tc) {
        std::cerr << "  ";
        tc->dump(std::cerr);
        std::cerr << "... ";

        auto &params = tc->getParams();
        ProcessingUnit pu(&progCtx, &params, &device, 1);

        {
            ProcessingUnit::PasswordWriter writer(pu);
            writer.setPassword(tc->getInput(), tc->getInputLength());
        }
        pu.beginProcessing();
        pu.endProcessing();

        ProcessingUnit::HashReader hash(pu);
        bool res = std::memcmp(tc->getOutput(), hash.getHash(),
                               params.getOutputLength()) == 0;
        if (!res) {
            ++failures;
            std::cerr << "FAIL" << std::endl;
        } else {
            std::cerr << "PASS" << std::endl;
        }
    }
    if (!failures) {
        std::cerr << "  ALL PASSED" << std::endl;
    }
    return failures;
}

const TestCase CASES_I_10[] = {
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 8, 1
        },
        "\xfd\x4d\xd8\x3d\x76\x2c\x49\xbd"
        "\xea\xf5\x7c\x47\xbd\xcd\x0c\x2f"
        "\x1b\xab\xf8\x63\xfd\xeb\x49\x0d"
        "\xf6\x3e\xde\x99\x75\xfc\xcf\x06",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 8, 2
        },
        "\xb6\xc1\x15\x60\xa6\xa9\xd6\x1e"
        "\xac\x70\x6b\x79\xa2\xf9\x7d\x68"
        "\xb4\x46\x3a\xa3\xad\x87\xe0\x0c"
        "\x07\xe2\xb0\x1e\x90\xc5\x64\xfb",
        "password", 8
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\xf6\xc4\xdb\x4a\x54\xe2\xa3\x70"
        "\x62\x7a\xff\x3d\xb6\x17\x6b\x94"
        "\xa2\xa2\x09\xa6\x2c\x8e\x36\x15"
        "\x27\x11\x80\x2f\x7b\x30\xc6\x94",
        "password", 8
    },
};

const TestCase CASES_I_13[] = {
    {
        {
            32,
            "\x02\x02\x02\x02\x02\x02\x02\x02"
            "\x02\x02\x02\x02\x02\x02\x02\x02", 16,
            "\x03\x03\x03\x03\x03\x03\x03\x03", 8,
            "\x04\x04\x04\x04\x04\x04\x04\x04"
            "\x04\x04\x04\x04", 12,
            3, 32, 4
        },
        "\xc8\x14\xd9\xd1\xdc\x7f\x37\xaa"
        "\x13\xf0\xd7\x7f\x24\x94\xbd\xa1"
        "\xc8\xde\x6b\x01\x6d\xd3\x88\xd2"
        "\x99\x52\xa4\xc4\x67\x2b\x6c\xe8",
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01"
        "\x01\x01\x01\x01\x01\x01\x01\x01", 32
    },
    {
        {
            32, "somesalt", 8, nullptr, 0, nullptr, 0,
            2, UINT32_C(1) << 16, 1
        },
        "\xc1\x62\x88\x32\x14\x7d\x97\x20"
        "\xc5\xbd\x1c\xfd\x61\x36\x70\x78"
        "\x72\x9f\x6d\xfb\x6f\x8f\xea\x9f"
        "\xf9\x81\x58\xe0\xd7\x81\x6e\xd0",
        "password", 8
    },
};

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define ARRAY_BEGIN(a) (a)
#define ARRAY_END(a) ((a) + ARRAY_SIZE(a))

int main(void) {
    std::size_t failures = 0;
    try {
        GlobalContext global;
        auto &devices = global.getAllDevices();
        auto &device = devices[0];

        failures += runTests(global, device, ARGON2_I, ARGON2_VERSION_10,
                             ARRAY_BEGIN(CASES_I_10), ARRAY_END(CASES_I_10));
        failures += runTests(global, device, ARGON2_I, ARGON2_VERSION_13,
                             ARRAY_BEGIN(CASES_I_13), ARRAY_END(CASES_I_13));
    } catch (cl::Error &err) {
        std::cerr << "OpenCL ERROR: " << err.err() << ": "
                  << err.what() << std::endl;
        return 2;
    }

    return !!failures;
}
