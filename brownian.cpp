//   Each particle undergoes 2D Brownian motion. The diffusion coefficient D
//   is derived from the Stokes-Einstein relation
//
// Output:
//   - CSV file with columns: step,time,particle_id,x,y
//   - Console printout of measured mean-squared-displacement (MSD) vs step,


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <map>
#include <chrono>
#include <algorithm>

struct Particle
{
    double x = 0.0;
    double y = 0.0;
    double x0 = 0.0; // initial position, kept to compute MSD
    double y0 = 0.0;
};

struct Config
{
    int numParticles = 200;
    long steps = 2000;
    double dt = 1.0e-4;      // seconds
    double temperature = 300.0; // kelvin
    double viscosity = 1.0e-3;  // Pa*s (water at room temp)
    double radius = 1.0e-6;     // meters (1 micron)
    unsigned long seed = 0;     // 0 => use random device
    std::string outFile = "trajectory.csv";
    long logEvery = 1;
    std::map<long, double> tempSchedule; // step -> new temperature
};

const double kB = 1.380649e-23; // Boltzmann constant, J/K

double diffusionCoefficient(double T, double eta, double r)
{
    return (kB * T) / (6.0 * M_PI * eta * r);
}

// parse "t1:T1,t2:T2" into a map<step, temperature>
std::map<long, double> parseSchedule(const std::string& s)
{
    std::map<long, double> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        auto pos = token.find(':');
        if (pos == std::string::npos) continue;
        long step = std::stol(token.substr(0, pos));
        double temp = std::stod(token.substr(pos + 1));
        result[step] = temp;
    }
    return result;
}

void printUsage()
{
    std::cout <<
    "Brownian motion particle simulator\n"
    "Usage: ./brownian [options]\n"
    "  --particles N          number of particles (default 200)\n"
    "  --steps N               number of timesteps (default 2000)\n"
    "  --dt SECONDS            timestep size (default 1e-4)\n"
    "  --temp KELVIN           temperature in Kelvin (default 300)\n"
    "  --viscosity PASCAL_S    fluid viscosity (default 1e-3, water)\n"
    "  --radius METERS         particle radius (default 1e-6, 1 micron)\n"
    "  --seed N                RNG seed (default: random)\n"
    "  --out FILE              CSV output path (default trajectory.csv)\n"
    "  --log-every N           write every Nth step to CSV (default 1)\n"
    "  --temp-schedule \"t1:T1,t2:T2\"  change temperature at given steps\n"
    "  --help                  show this message\n";
}

Config parseArgs(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string
        {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--particles") cfg.numParticles = std::stoi(next("--particles"));
        else if (arg == "--steps") cfg.steps = std::stol(next("--steps"));
        else if (arg == "--dt") cfg.dt = std::stod(next("--dt"));
        else if (arg == "--temp") cfg.temperature = std::stod(next("--temp"));
        else if (arg == "--viscosity") cfg.viscosity = std::stod(next("--viscosity"));
        else if (arg == "--radius") cfg.radius = std::stod(next("--radius"));
        else if (arg == "--seed") cfg.seed = std::stoul(next("--seed"));
        else if (arg == "--out") cfg.outFile = next("--out");
        else if (arg == "--log-every") cfg.logEvery = std::stol(next("--log-every"));
        else if (arg == "--temp-schedule") cfg.tempSchedule = parseSchedule(next("--temp-schedule"));
        else if (arg == "--help") { printUsage(); std::exit(0);
                                  }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            std::exit(1);
        }
    }
    return cfg;
}

int main(int argc, char** argv)
{
    Config cfg = parseArgs(argc, argv);

    unsigned long seed = cfg.seed;
    if (seed == 0)
    {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<Particle> particles(cfg.numParticles);
    // all particles start at the origin
    for (auto& p : particles)
      {
        p.x = p.y = 0.0;
        p.x0 = p.y0 = 0.0;
    }

    double T = cfg.temperature;
    double D = diffusionCoefficient(T, cfg.viscosity, cfg.radius);

    std::ofstream out(cfg.outFile);
    if (!out) {
        std::cerr << "Failed to open output file: " << cfg.outFile << "\n";
        return 1;
    }
    out << "step,time,particle_id,x,y,temperature\n";

    std::cout << "Brownian motion simulation\n"
              << "  particles   = " << cfg.numParticles << "\n"
              << "  steps       = " << cfg.steps << "\n"
              << "  dt          = " << cfg.dt << " s\n"
              << "  temperature = " << T << " K\n"
              << "  viscosity   = " << cfg.viscosity << " Pa*s\n"
              << "  radius      = " << cfg.radius << " m\n"
              << "  D (initial) = " << D << " m^2/s\n"
              << "  seed        = " << seed << "\n"
              << "  output      = " << cfg.outFile << "\n\n";

    std::cout << "step\ttime(s)\tT(K)\tD(m^2/s)\tmeasured_MSD(m^2)\ttheoretical_MSD(m^2)\n";

    // log step 0
    for (int i = 0; i < cfg.numParticles; ++i)
    {
        out << 0 << "," << 0.0 << "," << i << "," << particles[i].x << "," << particles[i].y << "," << T << "\n";
    }

    for (long step = 1; step <= cfg.steps; ++step)
    {
        // apply temperature schedule if this step has an entry
        auto it = cfg.tempSchedule.find(step);
        if (it != cfg.tempSchedule.end())
        {
            T = it->second;
            D = diffusionCoefficient(T, cfg.viscosity, cfg.radius);
        }

        double sigma = std::sqrt(2.0 * D * cfg.dt); // stddev of each displacement component

        for (auto& p : particles)
        {
            p.x += sigma * gauss(rng);
            p.y += sigma * gauss(rng);
        }

        if (step % cfg.logEvery == 0)
        {
            double t = step * cfg.dt;
            for (int i = 0; i < cfg.numParticles; ++i)
            {
                out << step << "," << t << "," << i << "," << particles[i].x << "," << particles[i].y << "," << T << "\n";
            }
        }

        if (step % std::max<long>(1, cfg.steps / 10) == 0 || step == cfg.steps)
        {
            double t = step * cfg.dt;
            double msdSum = 0.0;
            for (auto& p : particles)
            {
                double dx = p.x - p.x0;
                double dy = p.y - p.y0;
                msdSum += dx * dx + dy * dy;
            }
            double measuredMSD = msdSum / cfg.numParticles;
            double theoreticalMSD = 4.0 * D * t; // 2D: <r^2> = 4 * D * t
            std::cout << step << "\t" << t << "\t" << T << "\t" << D << "\t"
                      << measuredMSD << "\t" << theoreticalMSD << "\n";
        }
    }

    out.close();
    std::cout << "\nDone. Trajectories written to " << cfg.outFile << "\n";
    std::cout << "Tip: raise --temp and re-run to see D and the MSD growth rate increase.\n";
    return 0;
}
