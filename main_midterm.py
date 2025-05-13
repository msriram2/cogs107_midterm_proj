"""
Let:

N be the number of informants.
M be the number of items (questions).
Xij be the response of informant i to item j (0 or 1).
Zj be the latent "consensus" or "correct" answer for item j (0 or 1).
Di be the latent "competence" of informant i (probability of knowing the correct answer), where 0.5 ≤ Di ≤ 1.

"""

from setup import *  # Import common functions and libraries
import numpy as np 
import pandas as pd

#CHATGPT HELP: Figuring out how to open a csv file and append all values within dataset into a numpy array 
def open_dataset(): 
    plant_knowledge_pd = pd.read_csv('plant_knowledge.csv')
    plant_knowledge = plant_knowledge_pd.to_numpy() 

    N = len(plant_knowledge[:, 0])
    M = len(plant_knowledge[0])
    
    for row in plant_knowledge: 
        num_correct = #Create a dictionary of each participant's correct answers. Count the amount of correct answers each participant got
        for item in row: 
            if item == '1': 
                
         
    return plant_knowledge, N, M, Z

def pymc_model(data): 

    """
    The model assumes that an informant's response Xij depends on their competence Di and the consensus answer Zj. Specifically, 
    the probability that informant i gives the answer Zj for item j is Di. The probability they give the incorrect answer is 1 − Di.

    This can be formulated as a Bernoulli likelihood: Xij∼ Bernoulli(pij)

    Where the probability pij of informant i answering '1' for item j is: pij = Di if Zj = 1 pij = 1 − Di if Zj = 0

    This can be written more compactly as: pij = Zj × Di + (1 − Zj) × (1 − Di)
    """

    """For each informant's competence Di, choose a suitable prior distribution. Justify your choice in the report.
        - For Di, I will likely have to use an informative prior between 0.5 and 1"""

    #NOTE: Restructure the first function and try to organize the data for the rest
    #of this program

    #PRIORS 
    # For each informant's competence Di, I will be using an informative prior distribution 
    #   - Informative with the assumption that the informant will have some degree of prior knowledge
    #   about the subject based on cultural upbringing and assimilated knowledge  
    # For each consensus answer Zj, a beta prior will be used, as a result of the assumption 
    #   that the informant will have more prior knowledge than not on the subject. 
    #   - Using the Bernoulli distribution and a bernoulli prior assumping minimal prior 
    #       assumption. 

    with pm.Model as plant_knowledge_model(): 

        b = pm.beta('b', alpha = 6, beta = 4)
        D = pm.binomial("D", N=N, Z=Z, observed=N_correct) #Select priors 

        Z = pm.Bernoulli('Z', b, observed=obs)

        D_reshaped = D[:, None] 
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped) 

        #Connect observed data X to p. 

    