#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <random>
#include <functional>
#include <numeric>
#include <algorithm>
#include <getopt.h>

#include "Eigen/Eigen"


// typedef float real;
typedef double real;


template <class T>
using KernelFunction = std::function<T(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>&,
                                       const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>&)>;

static std::mt19937_64 rng;


// float kernel(const Eigen::Ref<const VectorXf> &x, const Eigen::Ref<const VectorXf> &y, float eps)
template <typename T>
T kernel(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& x,
         const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>& y, T eps)    
{
    if (x.size() != y.size())
        throw std::invalid_argument("x and y must have the same length");
    return exp( - (x - y).squaredNorm() / eps );
}



template <typename Derived, typename Derived2>
int construct_kernel_matrix(const Eigen::MatrixBase<Derived>& samples,
                            KernelFunction<typename Derived::Scalar> func,
                            Eigen::MatrixBase<Derived2>& matrix)
{
    unsigned long nsamples = samples.cols();
    unsigned long ndim = samples.rows();

    for (unsigned long col1 = 0; col1 < nsamples; col1++) {
        for (unsigned long col2 = col1; col2 < nsamples; col2++) {
            matrix(col2, col1) = func(samples.col(col1), samples.col(col2));
            matrix(col1, col2) = matrix(col2, col1);
        }
    }
    return 0;
}

template <typename Derived>
Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>>
diffusion_map(const Eigen::MatrixBase<Derived> & samples,
              KernelFunction<typename Derived::Scalar> func,
              real alpha)
{
    unsigned long num_samples = samples.cols();
    
    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic> kernel_matrix(num_samples, num_samples);
    construct_kernel_matrix(samples, func, kernel_matrix);
    
    auto density = kernel_matrix.rowwise().sum().array().pow(alpha).matrix();
    auto mask = density * density.transpose(); 
    auto ani_kernel = kernel_matrix.cwiseQuotient(mask);
    auto ani_density = ani_kernel.rowwise().sum();

    // We want to solve
    // D^{-1} K \phi = \lamdba phi
    // K \phi = \lambda D \phi
    auto D = ani_density.asDiagonal();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> es(ani_kernel, D);                                                         
    return es;
}

template <typename Derived>
std::vector<unsigned long> sorted_evals(const Eigen::MatrixBase<Derived>& eigenvals)
{
    /* Sort eigenvalues in decreasing order */
    std::vector<unsigned long> idx(eigenvals.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&eigenvals](size_t ii, size_t jj)
                         {
                             return std::abs(eigenvals[ii]) > std::abs(eigenvals[jj]);
                         });
    return idx;
}

static char * program_name;

void print_code_usage (FILE *, int) __attribute__ ((noreturn));
void print_code_usage (FILE * stream, int exit_code)
{

    // fprintf(stream, "\nUsage: %s <filename> \n\n", program_name);
    fprintf(stream,
            "\nSample Usage\n"
            "./diffusion_map [-t time] [-a alpha] [-e epsilon] [-o fileout] filein\n"
            
            "\n\n"

            "-a --alpha     <val>:   Anisotropy factor (default 1.0):\n"
            "                        0:   Laplacian point density matters\n"
            "                        0.5: Fokker-Planck\n"
            "                        1.0: Laplace-Beltrami, point density doesnt matter\n"
            "                              this value separates statistics from geometry\n"
            "-e --epsilon   <val>:   Kernel distance factor (default 1.0) exp(-||x - y||/epsilon)\n"
            "-t --time      <val>:   Diffusion time\n"
            "-n --numcoord  <val>:   Number of coordinates to print\n"
            "-o --output    <name>:  filename to write with the new coordinates\n"
        );
    exit (exit_code);
}

int main(int argc, char *argv[])
{
    program_name = argv[0];
    
    int next_option;
    const char * const short_options = "he:a:t:n:o:";
    const struct option long_options[] = {
        { "help"       ,  no_argument, nullptr, 'h' },
        { "epsilon"    ,  required_argument, nullptr, 'e' },
        { "alpha"      ,  required_argument, nullptr, 'a' },
        { "time "      ,  required_argument, nullptr, 't' },
        { "numcoord"   ,  required_argument, nullptr, 'n' },
        { "output"     ,  required_argument, nullptr, 'o' },
        { nullptr      ,  0, nullptr, 0   }
    };
    
    fprintf(stdout, "\n\n");
    fprintf(stdout, "****************************************\n");
    fprintf(stdout, "*******     Diffusion Maps     *********\n");
    fprintf(stdout, "*******     --------------     *********\n");
    fprintf(stdout, "****************************************\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "Called as: %s ", argv[0]);
    for (int ii = 1; ii < argc; ii++){
        fprintf(stdout, " %s ", argv[ii]);
    }
    fprintf(stdout, "\n");
    

    // Algorithm setup
    real eps = 1.0;
    real alpha = 1.0;
    real time = 0.0;
    unsigned num_keep = 10;

    std::string filename_out = "dmapcoord.dat";

    int num_opts = 0;    
    do {
        next_option = getopt_long (argc, argv, short_options, long_options, NULL);
        num_opts++;
        switch (next_option)
        {
        case 'h': 
            print_code_usage(stdout, 0);
        case 'e':
            eps = atof(optarg);
            break;
        case 'a':
            alpha = atof(optarg);
            break;
        case 't':
            time = atof(optarg);
            break;
        case 'n':
            num_keep = (unsigned)atoi(optarg);
            break;
        case 'o':
            filename_out = optarg;
            break;                        
        case '?': // The user specified an invalid option
            // printf("invalid option %s\n\n",optarg);
            print_code_usage (stderr, 1);
        case -1: // Done with options. 
            break;
        default: // Something unexpected
            abort();
        }

    } while (next_option != -1);

    if (optind >= argc) {
        std::cerr << "Expected a filename as last argument" << std::endl;
        print_code_usage (stderr, 1);
    }

    std::string filename(argv[optind]);
    
    std::vector<real> vals;
    unsigned long num_cols = 0;
    unsigned long num_rows = 0;
    
    std::ifstream file;    
    file.open(filename);
    if (file.is_open()){
        std::string line;
        short int cols_set = 0;
        while(std::getline(file, line, '\n')) {
            // std::cout << line << std::endl;
            std::stringstream ss(line);
            unsigned long check_cols = 0;
            while (!ss.fail()) {
                real f;
                ss >> f;
                if (!ss.fail()) {
                    // std::cout << std::setprecision(17) << "\t" << f << std::endl;
                    vals.push_back(f);
                    if (cols_set == 0) {
                        num_cols += 1;
                    }
                    else{
                        check_cols += 1;
                    }
                }
            }
            if (cols_set == 1) {
                if (check_cols != num_cols) {
                    throw std::logic_error("Must have equal rows in the input file");
                }
            }
            cols_set = 1;
            num_rows += 1;
        }
        file.close();
    }    
    else {
        std::cout << "Unable to open file " << filename << std::endl;
        return 1;
    }

    std::cout << std::endl;
    std::cout << "Options" << std::endl;
    std::cout << "================" << std::endl;
    std::cout << "alpha        = " << alpha << std::endl;
    std::cout << "epsilon      = " << eps << std::endl;
    std::cout << "time         = " << time << std::endl;
    std::cout << "filename in  = " << filename << std::endl;
    std::cout << "filename out = " << filename_out << std::endl; 
    std::cout << std::endl;

    // note num_cols in the input file is number of rows in the samples because of column major ordering
    Eigen::Map<Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>> samples(vals.data(), num_cols, num_rows); 

    std::cout << "Status: Finished reading data" << std::endl;;
        
    auto kfun = std::bind(&kernel<real>, std::placeholders::_1, std::placeholders::_2, eps);
    auto es = diffusion_map(samples, kfun, alpha);
    auto eigen_vals = es.eigenvalues();
    auto eigen_vecs = es.eigenvectors();
    auto idx_sorted = sorted_evals(eigen_vals);
    auto eigen_vals_advanced = eigen_vals.array().pow(time);
    std::vector<real> eigen_norms(num_keep);
    for (unsigned long ii = 0; ii < num_keep; ii++){
        eigen_norms[ii] = sqrt(eigen_vecs.col(idx_sorted[ii]).norm());
    }
    std::cout << "Status: Finished Computing Eigenvectors" << std::endl;;
    
    // printing
    unsigned long width = 5;
    unsigned long precision = 8;

    std::ofstream out;
    out.open(filename_out);
    if (out.is_open()){
        out.precision(precision);
        out.width(width);
        out.setf(std::ios_base::scientific, std::ios_base::floatfield);    
            
        for (unsigned long sample = 0; sample < samples.cols(); sample++)  {
            for (unsigned long jj = 0; jj < num_keep; jj++) {
                auto coordinate = eigen_vals_advanced(idx_sorted[jj]) * eigen_vecs(sample, idx_sorted[jj]) / eigen_norms[jj];
                out << coordinate << " ";
            }
            out << std::endl;
        }
        std::cout << "Status: Finished Writing Coordinates" << std::endl;;
        out.close();
    }
    else {
        std::cerr << "Cannot open output file: " << filename_out << std::endl;
    }

    std::cout << "Status: Done" << std::endl;;    
    
    return 0;
}



