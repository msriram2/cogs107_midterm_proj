

import numpy as np 
import pandas as pd
import arviz as az
import pymc as pm

"""
Below is an explanation and interpretation of the variables

Let:

N be the number of informants.
    - i represents the individual informant and their answers to all questions M 
        - Might be represented as a row
M be the number of items (questions).
    - j represents the individual questions that all participants have answered
        - Might be represented as a column 
Xij be the response of informant i to item j (0 or 1).
    - ij represents the specific question 
        - Might be represented as the one answer at the intersection of i and j
Zj be the latent "consensus" or "correct" answer for item j (0 or 1).
    - Zj represents the individual "consensus" or "correct" answers 
        - Represented along the column
Di be the latent "competence" of informant i (probability of knowing the correct answer), where 0.5 ≤ Di ≤ 1.
    - Di represents competence among individual participants 
        - Represented along the row

"""

#CHATGPT HELP: Figuring out how to open a csv file and append all values within dataset into a numpy array 
def open_dataset(): 
    """
    GOAL: Open a csv file convert it into a numpy file with while removing the informants label column.
    
    Arguments: None 
    Returns: Numpy array  
    """
    plant_knowledge_pd = pd.read_csv('plant_knowledge.csv')
    plant_knowledge_np = plant_knowledge_pd.to_numpy() 
    plant_knowledge = np.delete(plant_knowledge_np, 0, axis = 1)
                
    return plant_knowledge_np

def pymc_model(data): 

    """
    GOAL: Establish priors for D and Z given number of informants and questions. Calculcates the probability that an informant will 
    get an answer correct based on the 'consensus answer' 

    Arguments: The array from the previous function 
    Returns: p, or the probability
    """

    """
    PRIORS 

    For each informant's competence Di, I will be using a Beta prior distribution 
      - Informative with the assumption that the informant will have some degree of prior knowledge
      - Beta prior, because assumptions are not discrete and fall between 0.5 <= D_i <= 1
      - Assuming that the subject will base answers on cultural upbringing and assimilated knowledge  

    For each consensus answer Zj, I will use Bernoulli distribution, since informants are still asked to give 
      either 0 or 1 (binary values) as an answer

    """

    "Determines number of informants (N) and number of questions (M)"
    N = len(data[:, 0])
    M = len(data[0])

    with pm.Model() as model: 

        #CHATGPT: After determining prior and distribution, used AI to confirm priors and reshape D and Z.

        #_____PRIORS_________

        """Defining the priors given vector of size N and M"""
        D_raw = pm.Beta('D_raw', alpha = 6, beta = 4, shape = N)
        D = 0.5 + 0.5 * D_raw
        Z = pm.Bernoulli('Z', p = D, shape = M)

        #_______LIKELIHOOD_______

        """Defining the proability of p_ij."""
        D_reshaped = D[:, None] 
        Z_reshaped = Z[None, :]
        prob = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        return Z

def plant_data_sample(model):

    """
    GOAL: Calculate posterior distribution 

    Arguments: Proability
    Returns: Sample data
    """

    #CHATGPT: Cross checking between AI for validation and 'conditional.py' format for understanding
    with model: 
        plant_posterior_data = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.8)
        az.plot_trace(plant_posterior_data)
        az.summary(plant_posterior_data)

    return plant_posterior_data

def plant_data_analyis(posterior_data, array):
     
    #Convergence Diagnostics:

    #CHATGPT: Cross checking and valiation between AI and 'conditional.py'
    plant_data_summary = az.summary(posterior_data, var_names=['p', 'D', 'Z'], hdi_prob = 0.94)
    print(plant_data_summary)

    #Posterior Mean Competence D_i: The mean of the posterior distribution for each D_i

    #CHATGPT: Copied code from AI
    posterior_means_D = plant_data_summary['mean'].values 
    for i, d in enumerate(posterior_means_D): 
        print(f"Informant {i+1}: Posterior mean competence D_{i+1} = {d:.3f}")
    plant_data_average_D = az.plot_posterior(plant_data_summary, var_names=['D'])
    print(plant_data_average_D)

    #Estimate Consensus Answers

    posterior_means_Z_raw = plant_data_summary['mean'].values 
    posterior_means_Z = round(posterior_means_Z_raw)
    for i, d in enumerate(posterior_means_Z): 
        print(f"Informant {i+1}: Posterior mean competence Z_{i+1} = {d:.3f}")
    plant_data_average_Z = az.plot_posterior(plant_data_summary, var_names=['Z'])
    print(plant_data_average_Z)

    #Comparing with Naive Aggregation

    #CHATGPT: Copied code to understand how to calculate the majority vote
    majority_vote = (np.sum(array, axis=0) > (array.shape[0] / 2)).astype(int)
    print('The Majority Vote is:', majority_vote)


def run_analysis(): 

    array = open_dataset()
    Model = pymc_model(array)
    posterior = plant_data_sample(Model)
    plant_data_analysis(posterior, array)
    

if __name__ == '__main__': 
    run_analysis()
    
