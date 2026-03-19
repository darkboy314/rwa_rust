use rand::{Rng, RngExt};
use std::cmp::Ordering;

/// Genetic Algorithm implementation
pub struct GA {
    pop_size: usize,
    generation_count: usize,
    mutation_rate: f64,
}

impl GA {
    pub fn new(pop_size: usize, generation_count: usize, mutation_rate: f64) -> Self {
        GA {
            pop_size,
            generation_count,
            mutation_rate,
        }
    }

    /// Run genetic algorithm
    pub fn run<F, P>(
        &self,
        obj_func: F,
        penalty_func: Option<P>,
        p_range: &[(f64, f64)],
        m_range: &[(f64, f64)],
    ) -> (Vec<f64>, f64)
    where
        F: Fn(&[f64]) -> f64,
        P: Fn(&[f64]) -> f64,
    {
        let mut rng = rand::rng();

        // Generate initial population
        let mut population = self.generate_random_population(&mut rng, p_range);

        // Evolution loop
        for _gen in 0..self.generation_count {
            // Create offspring and mutate in one pass to avoid extra intermediate allocations.
            let offspring = self.crossover_and_mutate(&population, &mut rng, m_range);

            // Combine and evaluate
            population.extend(offspring);

            let mut scored_population: Vec<(Vec<f64>, f64)> = population
                .into_iter()
                .map(|individual| {
                    let penalty = self.penalty_function(&individual, p_range);
                    let obj = obj_func(&individual);
                    let p_penalty = if let Some(p_func) = penalty_func.as_ref() {
                        p_func(&individual)
                    } else {
                        0.0
                    };

                    (individual, 2000.0 - obj - penalty - p_penalty)
                })
                .collect();

            scored_population.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            });
            scored_population.truncate(self.pop_size);

            population = scored_population
                .into_iter()
                .map(|(individual, _)| individual)
                .collect();
        }

        // Return best individual
        let best = population.swap_remove(0);

        // Check if penalty function is violated (constraint check)
        // Using a small epsilon for numerical stability
        let epsilon_constraint: f64 = 1e-6;
        if let Some(p_func) = penalty_func {
            if p_func(&best) > epsilon_constraint {
                let nan_vec = vec![f64::NAN; best.len()];
                return (nan_vec, f64::NAN);
            }
        }

        // Check if bounds are violated (boundary check)
        let epsilon_bound: f64 = 1e-6;
        let bound_penalty = self.penalty_function(&best, p_range);
        if bound_penalty > epsilon_bound {
            let nan_vec = vec![f64::NAN; best.len()];
            return (nan_vec, f64::NAN);
        }

        let best_fitness = obj_func(&best);
        (best, best_fitness)
    }

    fn generate_random_population(
        &self,
        rng: &mut impl Rng,
        p_range: &[(f64, f64)],
    ) -> Vec<Vec<f64>> {
        let mut population = Vec::with_capacity(self.pop_size);

        for _ in 0..self.pop_size {
            let individual: Vec<f64> = p_range
                .iter()
                .map(|(min, max)| rng.random_range(*min..*max))
                .collect();
            population.push(individual);
        }

        population
    }

    fn crossover_and_mutate(
        &self,
        population: &[Vec<f64>],
        rng: &mut impl Rng,
        m_range: &[(f64, f64)],
    ) -> Vec<Vec<f64>> {
        let mut offspring = Vec::with_capacity(population.len() / 2);

        for i in (0..population.len()).step_by(2) {
            if i + 1 < population.len() {
                let p1 = &population[i];
                let p2 = &population[i + 1];

                let child: Vec<f64> = p1
                    .iter()
                    .zip(p2.iter())
                    .zip(m_range.iter())
                    .map(|((a, b), (min_mut, max_mut))| {

                        let min = a.min(*b);
                        let range = (a - b).abs();
                        let mut gene = min + rng.random::<f64>() * range;

                        if rng.random::<f64>() < self.mutation_rate {
                            gene += rng.random_range(*min_mut..*max_mut);
                        }

                        gene
                    })
                    .collect();

                offspring.push(child);
            }
        }

        offspring
    }

    fn penalty_function(&self, individual: &[f64], bounds: &[(f64, f64)]) -> f64 {
        const PENALTY_WEIGHT: f64 = 1e100;
        let mut penalty = 0.0;

        for (value, (min_bound, max_bound)) in individual.iter().zip(bounds.iter()) {
            if value < min_bound {
                penalty += PENALTY_WEIGHT * (min_bound - value).powi(2);
            } else if value > max_bound {
                penalty += PENALTY_WEIGHT * (value - max_bound).powi(2);
            }
        }

        penalty
    }

}

#[cfg(test)]
mod tests {
    use super::GA;

    fn eggholder(x: &[f64]) -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        -(x2 + 47.0) * ((x2 + x1 / 2.0 + 47.0).abs().sqrt()).sin()
            - x1 * ((x1 - (x2 + 47.0)).abs().sqrt()).sin()
    }

    #[test]
    fn ga_can_optimize_eggholder() {
        let ga = GA::new(600, 350, 0.15);
        let p_range = [(-512.0, 512.0), (-512.0, 512.0)];
        let m_range = [(-20.0, 20.0), (-20.0, 20.0)];

        let mut best_value = f64::INFINITY;
        for _ in 0..10 {
            let (_best_x, value) = ga.run(
                eggholder,
                None::<fn(&[f64]) -> f64>,
                &p_range,
                &m_range,
            );
            if value < best_value {
                best_value = value;
            }
        }

        println!("best Eggholder value over 10 runs: {best_value}");

        // Global optimum is about -959.64; with stochastic GA we use a robust threshold.
        assert!(best_value < -850.0, "best Eggholder value: {}", best_value);
    }
}
