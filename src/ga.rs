use rand::{Rng, RngExt};

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
            // Create offspring through crossover
            let mut offspring = self.crossover(&population);

            // Apply mutation
            offspring = self.mutate(&offspring, &mut rng, m_range);

            // Combine and evaluate
            population.extend(offspring);
            let scores = self.evaluate(&population, &obj_func, penalty_func.as_ref(), p_range);

            // Select best individuals
            population = self.filter(&population, &scores, self.pop_size);

            if _gen % 100 == 0 {
                println!("Generation {}: best fitness = {}", _gen, scores[0]);
            }
        }

        // Return best individual
        let best = population[0].clone();

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
        let mut population = Vec::new();

        for _ in 0..self.pop_size {
            let individual: Vec<f64> = p_range
                .iter()
                .map(|(min, max)| rng.random_range(*min..*max))
                .collect();
            population.push(individual);
        }

        population
    }

    fn crossover(&self, population: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut rng = rand::rng();
        let mut offspring = Vec::new();

        for i in (0..population.len()).step_by(2) {
            if i + 1 < population.len() {
                let p1 = &population[i];
                let p2 = &population[i + 1];

                let child: Vec<f64> = p1
                    .iter()
                    .zip(p2.iter())
                    .map(|(a, b)| {
                        let min = a.min(*b);
                        let range = (a - b).abs();
                        min + rng.random::<f64>() * range
                    })
                    .collect();

                offspring.push(child);
            }
        }

        offspring
    }

    fn mutate(
        &self,
        population: &[Vec<f64>],
        rng: &mut impl Rng,
        m_range: &[(f64, f64)],
    ) -> Vec<Vec<f64>> {
        population
            .iter()
            .map(|individual| {
                individual
                    .iter()
                    .zip(m_range.iter())
                    .map(|(gene, (min, max))| {
                        if rng.random::<f64>() < self.mutation_rate {
                            gene + rng.random_range(*min..*max)
                        } else {
                            *gene
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn evaluate<F, P>(
        &self,
        population: &[Vec<f64>],
        obj_func: &F,
        penalty_func: Option<&P>,
        bounds: &[(f64, f64)],
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
        P: Fn(&[f64]) -> f64,
    {
        population
            .iter()
            .map(|individual| {
                let penalty = self.penalty_function(individual, bounds);
                let obj = obj_func(individual);
                let p_penalty = if let Some(p_func) = penalty_func {
                    p_func(individual)
                } else {
                    0.0
                };

                2000.0 - obj - penalty - p_penalty
            })
            .collect()
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

    fn filter(&self, population: &[Vec<f64>], scores: &[f64], top_n: usize) -> Vec<Vec<f64>> {
        let mut indexed: Vec<(usize, f64)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        indexed
            .iter()
            .take(top_n)
            .map(|(idx, _)| population[*idx].clone())
            .collect()
    }
}
