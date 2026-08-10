import tools.moea_fs as moea_fs
import graph_tool.all as gt
import pandas as pd, numpy as np 
import time , logging
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    filename = "/home/cursed/Documenti/Results_MOBIUS/GOVEGAN/Cohorts_Data/data_for_ML/feature_graph.gt"
    network = gt.load_graph(filename)

    print("\n\n\n")

    print(network)

    print(list(network.vp.keys()))

    g_communities = pd.read_csv("graph_communities.csv", index_col=0, header=0)
    print(g_communities)

    net_problem = moea_fs.NetworkMOEAProblem(
        g = network,
        df_communities=g_communities, 
        cd_method = "Louvain",
        score = "vscore_AUC" 
    )

    net_problem.set_relevance_function(func = "score", score="vscore_AUC" )


    initial_sample = moea_fs.BinaryRandomSampling() 
    algorithm = moea_fs.NSGA2(
        pop_size=60,
        sampling=initial_sample,
        crossover= moea_fs.ExponentialCrossover(), #HalfUniformCrossover(),#UniformCrossover(),#TwoPointCrossover(),
        mutation=moea_fs.BitflipMutation(prob = 0.1),
        eliminate_duplicates=True )
    termination = moea_fs.RobustTermination(
        moea_fs.MultiObjectiveSpaceTermination( n_skip=5 ) ) 
    
    algorithm.setup(
            net_problem, 
            termination=termination, 
            seed=42, 
            verbose=False, 
            save_history=True)
        

    max_n_gen = 1000
    time_spent = time_gen = time.time()

    while algorithm.has_next():
            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()

            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(net_problem, pop)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

            
            # do same more things, printing, logging, storing or even modifying the algorithm object
            if algorithm.n_gen % 50 == 0:
                mean_rel, mean_red = np.mean(list(map(lambda x: x._F, pop)), axis=0)
                logging.critical(f"Generation {algorithm.n_gen} -- n_eval: {algorithm.evaluator.n_eval} -- avg. rel: {mean_rel:.3f} -- avg. red: {mean_red:.3f} -- req. time: {(time.time() - time_gen):.2f} seconds")
                time_gen = time.time()

            if algorithm.n_gen > max_n_gen:
                break
        
    mean_rel, mean_red = np.mean(list(map(lambda x: x._F, pop)), axis=0)
    time_spent = time.time() - time_spent
    print(f"Final generation {algorithm.n_gen} -- n_eval: {algorithm.evaluator.n_eval} -- avg. rel: {mean_rel:.3f} -- avg. red: {mean_red:.3f}")
    print(f"Total time: {time_spent:.2f} seconds -- avg. time per generation: {(time.time() - time_gen)/algorithm.n_gen:.2f} seconds")

    results = algorithm.results()
    print("We terminated, motherfucker!\n\n")
    print(results)

