#include "gbo.h"


static arma::vec calculate_gsr(double rho2, const arma::vec& best_x, const arma::vec& worst_x, const arma::vec& current_x, const arma::vec& xr1, const arma::vec& dm, const arma::vec& xm, unsigned char flag, int N){
    int vec_size = best_x.n_elem;
    arma::vec gsr(vec_size), del_x(vec_size), delta(vec_size), step(vec_size), xs(vec_size), yp(vec_size), yq(vec_size);
    double a = uniform_random_0_1();
    double b = static_cast<double>(random_index(N));
    double c = uniform_random_0_1();
    double eps =  0.01 * uniform_random_0_1();

    delta = 2.0 * a * arma::abs(xm - current_x + eps);
    step = 0.5 * (best_x - xr1 + delta);
    del_x = b * arma::abs(step);
    gsr = (c * rho2 * 2.0 * (del_x % current_x)) / (best_x - worst_x + eps);
    xs = flag == 1? current_x : best_x;
    xs = xs - gsr + dm;

    double p1 = uniform_random_0_1();
    double p2 = uniform_random_0_1();
    double q1 = uniform_random_0_1();
    double q2 = uniform_random_0_1();
    double d = gaussian_random_0_1();

    yp = p1 * (0.5 * (xs + current_x) + p2 * del_x);
    yq = q1 * (0.5 * (xs + current_x) - q2 * del_x);
    gsr = (d * rho2 * 2.0 * (del_x % current_x)) / (yp - yq + eps);
    

    return gsr;
}

cv::Mat GBO::main_loop(cv::Mat& block, int vector_size, unsigned char bit, int scheme) {
    Population population(vector_size, block, bit, scheme);
    for (int m = 0; m < GBO::iterations; ++m) {
        double betta = GBO::betta_min + (GBO::betta_max - GBO::betta_min) * std::pow(1.0 - std::pow(static_cast<double>(m + 1) / static_cast<double>(GBO::iterations), 3.0), 2.0);
        double alpha = std::fabs(betta * std::sin(GBO::angle + std::sin(GBO::angle * betta)));

        for (int current_vector = 0; current_vector < population.individuals.size(); ++current_vector) {
            double rho1 = alpha * (2.0 * uniform_random_0_1() - 1.0);
            double rho2 = alpha * (2.0 * uniform_random_0_1() - 1.0);
            double dm_rand = uniform_random_0_1();
            std::vector<int> random_indices = generate_random_indices(population.individuals.size(), population.indexOfBestIndividual, current_vector);

            arma::vec x1(vector_size), x2(vector_size), x3(vector_size), xm(vector_size), dm(vector_size), gsr(vector_size), x_next(vector_size);

            xm = 0.25 * (population.individuals[random_indices[0]] + population.individuals[random_indices[1]] + population.individuals[random_indices[2]] + population.individuals[random_indices[3]]);
            dm = dm_rand * rho1 * (population.individuals[population.indexOfBestIndividual] - population.individuals[random_indices[0]]);
            gsr = calculate_gsr(rho2, population.individuals[population.indexOfBestIndividual], population.worstIndividual, population.individuals[current_vector], population.individuals[random_indices[0]], dm, xm, 1, population.individuals.size());

            dm_rand = uniform_random_0_1();
            dm = dm_rand * rho1 * (population.individuals[population.indexOfBestIndividual] - population.individuals[random_indices[0]]);
            x1 = population.individuals[current_vector] + dm - gsr;

            dm_rand = uniform_random_0_1();
            dm = dm_rand * rho1 * (population.individuals[random_indices[0]] - population.individuals[random_indices[1]]);
            gsr = calculate_gsr(rho2, population.individuals[population.indexOfBestIndividual], population.worstIndividual, population.individuals[current_vector], population.individuals[random_indices[0]], dm, xm, 2, population.individuals.size());

            dm_rand = uniform_random_0_1();
            dm = dm_rand * rho1 * (population.individuals[random_indices[0]] - population.individuals[random_indices[1]]);
            x2 = population.individuals[population.indexOfBestIndividual] + dm - gsr;

            rho1 = alpha * (2.0 * uniform_random_0_1() - 1.0);
            double ra = uniform_random_0_1();
            double rb = uniform_random_0_1();

            x3 = population.individuals[current_vector] - rho1 * (x2 - x1);
            x_next = ra * (rb * x1 + (1 - rb) * x2) + (1 - ra) * x3;
            x_next.clamp(-1.0 * th, th);

            //LEO

            if (uniform_random_0_1() < PR) {
                double L1 = (uniform_random_0_1() < 0.5) ? 0.0 : 1.0;
                double u1 = L1 * 2.0 * uniform_random_0_1() + (1.0 - L1);
                double u2 = L1 * uniform_random_0_1() + (1.0 - L1);
                double u3 = L1 * uniform_random_0_1() + (1.0 - L1);
                double nu2 = uniform_random_0_1();

                arma::vec x_mk(vector_size), x_p(vector_size), x_rand(vector_size), Y(vector_size);
                x_p = population.individuals[random_index(population.individuals.size())];
                x_rand.randu(vector_size);
                x_rand = 2.0 * th * x_rand - th;

                double L2 = (uniform_random_0_1() < 0.5) ? 0.0 : 1.0;

                x_mk = L2 * x_p + (1.0 - L2) * x_rand;
                Y = uniform_random_0_1() < 0.5 ? x_next : population.individuals[population.indexOfBestIndividual];

                double f1 = 2.0 * uniform_random_0_1() - 1.0;
                double f2 = 2.0 * uniform_random_0_1() - 1.0;

                x_next = Y + f1 * (u1 * population.individuals[population.indexOfBestIndividual] - u2 * x_mk) + f2 * rho1 * (u3 * (x2 -x1) + u2 * (population.individuals[random_indices[0]] - population.individuals[random_indices[1]])) * 0.5;
                x_next.clamp(-1.0 * th, th);
            }

            population.update(x_next, current_vector);
            
        }
    }
    cv::Mat result_block = applyVectorToBlock(population.individuals[population.indexOfBestIndividual], block, scheme);
    return result_block;
}