import random
import numpy as np
import pandas as pd


class ga:
    def __init__(self):
        pass

    @staticmethod
    def run(
        obj_func,
        *,
        p_func=None,
        pop_size=1000,
        m_rate=0.1,
        gen_n=1000,
        p_range,
        m_range,
    ):
        # Generate initial population
        population = ga.generate_random_population(pop_size, *p_range)
        scores = ga.evaluate(population, obj_func, p_func, *p_range)
        offspring = np.zeros((len(population) // 2, len(p_range)), dtype=float)

        for _ in range(gen_n):  # Number of generations
            # Shuffle the population to ensure randomness
            random.shuffle(population)

            # Create a new population using crossovers
            offspring = ga.crossover(population)

            # mutation
            offspring = ga.mutation(offspring, m_rate, *m_range)

            # Evaluate the new population
            population = np.concatenate((population, offspring), axis=0)
            scores = ga.evaluate(population, obj_func, p_func, *p_range)

            # Filter the population based on scores
            population = ga.filtering(population, scores, top_n=pop_size)

            # print(f"{_}th gen population:{population[0]}")
            # print(f"{_}th final score:{obj_func(*population[0])}")

        if p_func(*population[0]) > 0:
            return np.full(np.shape(population[0]), np.nan), np.nan

        if ga.penalty_function_nd(population[0], p_range) > 0:
            return np.full(np.shape(population[0]), np.nan), np.nan

        return population[0], obj_func(*population[0])

    @staticmethod
    def generate_random_population(population_size, *range_args):
        pop = np.concatenate(
            [
                np.random.uniform(arg[0], arg[1], population_size).reshape(-1, 1)
                for arg in range_args
            ],
            axis=1,
        )
        return pop

    @staticmethod
    def crossover(pop):
        pop1 = pop[::2]
        pop2 = pop[1::2]
        min = np.minimum(pop1, pop2)
        rnd = np.random.rand(*pop1.shape)
        offspring = min + rnd * (np.abs(pop1 - pop2))
        return offspring

    @staticmethod
    def mutation(pop, mutation_rate, *range_args):
        bool_array = np.random.rand(len(pop), len(range_args)) < mutation_rate
        return pop + bool_array * np.random.uniform(*np.array(range_args).T)

    @staticmethod
    def evaluate(pop, obj_func, p_func, *bound):
        p = np.array([ga.penalty_function_nd(ind, bound) for ind in pop])
        if p_func is not None:
            p2 = np.array([p_func(*ind) for ind in pop])
            return 2000 - np.array([obj_func(*ind) for ind in pop]) - p - p2
        return 2000 - np.array([obj_func(*ind) for ind in pop]) - p

    @staticmethod
    def filtering(population, scores, top_n=100):
        i = np.argsort(scores)[::-1]
        sorted = population[i]
        return sorted[:top_n]

    @staticmethod
    def penalty_function_nd(individual, bounds, penalty_weight=1e100):
        penalty = 0.0
        for i, (value, (min_bound, max_bound)) in enumerate(zip(individual, bounds)):
            if value < min_bound:
                penalty += penalty_weight * (min_bound - value) ** 2
            elif value > max_bound:
                penalty += penalty_weight * (value - max_bound) ** 2
        return penalty
