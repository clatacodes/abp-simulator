
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
#include <numeric>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Particle
{
    double x = 0.0, y = 0.0;    // wrapped position, in [0, L)
    double ux = 0.0, uy = 0.0;  // unwrapped position (for MSD)
    double ux0 = 0.0, uy0 = 0.0;
    double theta = 0.0;         // heading angle (rad)
    double fx = 0.0, fy = 0.0;  // interaction force (N)
};

struct Config
{
    int numParticles = 400;
    long steps = 200000;
    double dt = 1.0e-3;          // seconds
    double temperature = 300.0;  // kelvin
    double viscosity = 1.0e-3;   // Pa*s
    bool waterViscosity = false; // use eta(T) for water instead of fixed viscosity
    double radius = 1.0e-6;      // meters
    double v0 = 3.0e-6;          // self-propulsion speed, m/s
    double phi = 0.5;            // area packing fraction
    double epsKT = 0.0;          // WCA epsilon in units of kT; 0 => automatic
    unsigned long seed = 0;      // 0 => random device
    std::string outFile = "trajectory.csv";
    long logEvery = 1000;
    std::map<long, double> tempSchedule; // step -> new temperature
};

const double kB = 1.380649e-23; // J/K

// Vogel equation for the viscosity of water (Pa*s), valid ~ 273-373 K
double waterEta(double T)
{
    return 2.414e-5 * std::pow(10.0, 247.8 / (T - 140.0));
}

std::map<long, double> parseSchedule(const std::string& s)
{
    std::map<long, double> result;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        auto pos = token.find(':');
        if (pos == std::string::npos) continue;
        result[std::stol(token.substr(0, pos))] = std::stod(token.substr(pos + 1));
    }
    return result;
}

void printUsage()
{
    std::cout <<
    "Active Brownian particle simulator\n"
    "Usage: ./abp [options]\n"
    "  --particles N          number of particles (default 400)\n"
    "  --steps N              number of timesteps (default 200000)\n"
    "  --dt SECONDS           timestep (default 1e-3)\n"
    "  --temp KELVIN          temperature (default 300)\n"
    "  --viscosity PASCAL_S   fluid viscosity (default 1e-3)\n"
    "  --water-eta            use water viscosity eta(T) instead of --viscosity\n"
    "  --radius METERS        particle radius (default 1e-6)\n"
    "  --v0 M_PER_S           self-propulsion speed (default 3e-6; 0 = passive)\n"
    "  --phi FRACTION         area packing fraction, < 0.75 (default 0.5)\n"
    "  --eps KT               WCA epsilon in units of kT (default: automatic)\n"
    "  --seed N               RNG seed (default: random)\n"
    "  --out FILE             CSV output path (default trajectory.csv)\n"
    "  --log-every N          write every Nth step (default 1000)\n"
    "  --temp-schedule \"s1:T1,s2:T2\"  change temperature at given steps\n"
    "  --help                 show this message\n";
}

Config parseArgs(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string
        {
            if (i + 1 >= argc)
            {
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
        else if (arg == "--water-eta") cfg.waterViscosity = true;
        else if (arg == "--radius") cfg.radius = std::stod(next("--radius"));
        else if (arg == "--v0") cfg.v0 = std::stod(next("--v0"));
        else if (arg == "--phi") cfg.phi = std::stod(next("--phi"));
        else if (arg == "--eps") cfg.epsKT = std::stod(next("--eps"));
        else if (arg == "--seed") cfg.seed = std::stoul(next("--seed"));
        else if (arg == "--out") cfg.outFile = next("--out");
        else if (arg == "--log-every") cfg.logEvery = std::stol(next("--log-every"));
        else if (arg == "--temp-schedule") cfg.tempSchedule = parseSchedule(next("--temp-schedule"));
        else if (arg == "--help") { printUsage(); std::exit(0); }
        else
        {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage();
            std::exit(1);
        }
    }
    return cfg;
}

// Transport coefficients at temperature T
struct Transport
{
    double eta, gamma, Dt, Dr;
};

Transport computeTransport(const Config& cfg, double T)
{
    Transport t;
    t.eta = cfg.waterViscosity ? waterEta(T) : cfg.viscosity;
    t.gamma = 6.0 * M_PI * t.eta * cfg.radius;
    t.Dt = kB * T / t.gamma;
    t.Dr = kB * T / (8.0 * M_PI * t.eta * std::pow(cfg.radius, 3));
    return t;
}

// Minimum-image separation component
inline double minImage(double d, double L)
{
    return d - L * std::round(d / L);
}

int main(int argc, char** argv)
{
    Config cfg = parseArgs(argc, argv);

    if (cfg.phi <= 0.0 || cfg.phi >= 0.75)
    {
        std::cerr << "--phi must be in (0, 0.75) for the lattice start.\n";
        return 1;
    }

    unsigned long seed = cfg.seed;
    if (seed == 0)
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 2.0 * M_PI);

    const int N = cfg.numParticles;
    const double sigma = 2.0 * cfg.radius;              // particle diameter
    const double L = std::sqrt(N * M_PI * cfg.radius * cfg.radius / cfg.phi);
    const double rcut = std::pow(2.0, 1.0 / 6.0) * sigma; // WCA cutoff
    const double rcut2 = rcut * rcut;

    double T = cfg.temperature;
    Transport tr = computeTransport(cfg, T);

    // WCA strength: automatic value is large enough that the active force
    // v0*gamma cannot push particles deep into each other.
    double Fa = cfg.v0 * tr.gamma;
    double eps = (cfg.epsKT > 0.0) ? cfg.epsKT * kB * T
                                   : std::max(kB * T, Fa * sigma / 24.0);

    // ---- Initial condition: randomly chosen sites of a square lattice, random headings
    int nSide = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N))));
    double a = L / nSide;
    if (a < sigma)
    {
        std::cerr << "Lattice spacing < particle diameter; lower --phi or --particles.\n";
        return 1;
    }
    std::vector<int> sites(nSide * nSide);
    std::iota(sites.begin(), sites.end(), 0);
    std::shuffle(sites.begin(), sites.end(), rng);

    std::vector<Particle> p(N);
    for (int i = 0; i < N; ++i)
    {
        int s = sites[i];
        p[i].x = (s % nSide + 0.5) * a;
        p[i].y = (s / nSide + 0.5) * a;
        p[i].ux = p[i].ux0 = p[i].x;
        p[i].uy = p[i].uy0 = p[i].y;
        p[i].theta = uni(rng);
    }

    // ---- Cell list
    int nc = static_cast<int>(std::floor(L / rcut));
    bool useCells = (nc >= 3);
    double cellSize = useCells ? L / nc : L;
    std::vector<int> head(useCells ? nc * nc : 1), nextIdx(N);

    // ---- Output
    std::ofstream out(cfg.outFile);
    if (!out)
    {
        std::cerr << "Failed to open output file: " << cfg.outFile << "\n";
        return 1;
    }
    out << "step,time,particle_id,x,y,temperature,theta\n";

    {
        std::ofstream meta(cfg.outFile + ".params.txt");
        meta << "box_L_m=" << L << "\n"
             << "particle_diameter_m=" << sigma << "\n"
             << "numParticles=" << N << "\n"
             << "phi=" << cfg.phi << "\n"
             << "v0_m_per_s=" << cfg.v0 << "\n"
             << "dt_s=" << cfg.dt << "\n"
             << "seed=" << seed << "\n";
    }

    double Pe = (tr.Dt > 0) ? cfg.v0 * sigma / tr.Dt : 0.0;

    std::cout << "Active Brownian particle simulation\n"
              << "  particles   = " << N << "\n"
              << "  steps       = " << cfg.steps << "  (total " << cfg.steps * cfg.dt << " s)\n"
              << "  dt          = " << cfg.dt << " s\n"
              << "  temperature = " << T << " K\n"
              << "  viscosity   = " << tr.eta << " Pa*s\n"
              << "  radius      = " << cfg.radius << " m  (sigma = " << sigma << " m)\n"
              << "  v0          = " << cfg.v0 << " m/s\n"
              << "  phi         = " << cfg.phi << "  (box L = " << L << " m)\n"
              << "  Dt          = " << tr.Dt << " m^2/s\n"
              << "  Dr          = " << tr.Dr << " 1/s\n"
              << "  Pe          = " << Pe << "\n"
              << "  WCA eps     = " << eps / (kB * T) << " kT\n"
              << "  cell list   = " << (useCells ? "on" : "off (box too small, O(N^2))") << "\n"
              << "  seed        = " << seed << "\n"
              << "  output      = " << cfg.outFile << "\n\n";

    // Stability warnings
    if (cfg.v0 * cfg.dt > 0.01 * sigma)
        std::cout << "WARNING: v0*dt > 0.01*sigma; consider a smaller --dt.\n";
    if (std::sqrt(2.0 * tr.Dt * cfg.dt) > 0.05 * sigma)
        std::cout << "WARNING: thermal step > 0.05*sigma; consider a smaller --dt.\n";
    if (std::sqrt(2.0 * tr.Dr * cfg.dt) > 0.1)
        std::cout << "WARNING: rotational step > 0.1 rad; consider a smaller --dt.\n";

    std::cout << "step\ttime(s)\tT(K)\tmeasured_MSD(m^2)\tfree_ABP_MSD(m^2)\n";

    auto writeFrame = [&](long step)
    {
        double t = step * cfg.dt;
        for (int i = 0; i < N; ++i)
            out << step << "," << t << "," << i << "," << p[i].x << "," << p[i].y
                << "," << T << "," << p[i].theta << "\n";
    };
    writeFrame(0);

    // Pair force accumulation
    auto addPair = [&](int i, int j)
    {
        double dx = minImage(p[i].x - p[j].x, L);
        double dy = minImage(p[i].y - p[j].y, L);
        double r2 = dx * dx + dy * dy;
        if (r2 >= rcut2 || r2 == 0.0) return;
        double sr2 = sigma * sigma / r2;
        double sr6 = sr2 * sr2 * sr2;
        double fOverR = 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r2;
        p[i].fx += fOverR * dx;  p[i].fy += fOverR * dy;
        p[j].fx -= fOverR * dx;  p[j].fy -= fOverR * dy;
    };

    for (long step = 1; step <= cfg.steps; ++step)
    {
        // temperature schedule
        auto it = cfg.tempSchedule.find(step);
        if (it != cfg.tempSchedule.end())
        {
            T = it->second;
            tr = computeTransport(cfg, T);
        }

        // ---- forces
        for (auto& q : p) q.fx = q.fy = 0.0;

        if (useCells)
        {
            std::fill(head.begin(), head.end(), -1);
            for (int i = 0; i < N; ++i)
            {
                int cx = std::min(nc - 1, static_cast<int>(p[i].x / cellSize));
                int cy = std::min(nc - 1, static_cast<int>(p[i].y / cellSize));
                int c = cy * nc + cx;
                nextIdx[i] = head[c];
                head[c] = i;
            }
            for (int i = 0; i < N; ++i)
            {
                int cx = std::min(nc - 1, static_cast<int>(p[i].x / cellSize));
                int cy = std::min(nc - 1, static_cast<int>(p[i].y / cellSize));
                for (int oy = -1; oy <= 1; ++oy)
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        int nx = (cx + ox + nc) % nc;
                        int ny = (cy + oy + nc) % nc;
                        for (int j = head[ny * nc + nx]; j != -1; j = nextIdx[j])
                            if (j > i) addPair(i, j);
                    }
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
                for (int j = i + 1; j < N; ++j)
                    addPair(i, j);
        }

        // ---- integrate (Euler-Maruyama)
        const double noiseT = std::sqrt(2.0 * tr.Dt * cfg.dt);
        const double noiseR = std::sqrt(2.0 * tr.Dr * cfg.dt);
        for (auto& q : p)
        {
            double dx = (cfg.v0 * std::cos(q.theta) + q.fx / tr.gamma) * cfg.dt + noiseT * gauss(rng);
            double dy = (cfg.v0 * std::sin(q.theta) + q.fy / tr.gamma) * cfg.dt + noiseT * gauss(rng);
            q.ux += dx;  q.uy += dy;
            q.x += dx;   q.y += dy;
            q.x -= L * std::floor(q.x / L);
            q.y -= L * std::floor(q.y / L);
            q.theta += noiseR * gauss(rng);
        }

        if (step % cfg.logEvery == 0)
            writeFrame(step);

        if (step % std::max<long>(1, cfg.steps / 10) == 0 || step == cfg.steps)
        {
            double t = step * cfg.dt;
            double msdSum = 0.0;
            for (auto& q : p)
            {
                double dx = q.ux - q.ux0, dy = q.uy - q.uy0;
                msdSum += dx * dx + dy * dy;
            }
            double measured = msdSum / N;

            std::cout << step << "\t" << t << "\t" << T << "\t" << measured << "\t";
            if (cfg.tempSchedule.empty())
            {
                double tau = 1.0 / tr.Dr;
                double theory = 4.0 * tr.Dt * t
                              + 2.0 * cfg.v0 * cfg.v0 * tau * (t - tau * (1.0 - std::exp(-t / tau)));
                std::cout << theory << "\n";
            }
            else
                std::cout << "-\n";
        }
    }

    out.close();
    std::cout << "\nDone. Trajectories written to " << cfg.outFile << "\n"
              << "Box size and parameters written to " << cfg.outFile << ".params.txt\n"
              << "Note: free-ABP MSD only matches at low phi; at high phi crowding lowers the measured MSD.\n";
    return 0;
}
