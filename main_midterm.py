"""
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
import numpy as np 
import pandas as pd

#CHATGPT HELP: Figuring out how to open a csv file and append all values within dataset into a numpy array 
def open_dataset(): 
    plant_knowledge_pd = pd.read_csv('plant_knowledge.csv')
    plant_knowledge_np = plant_knowledge_pd.to_numpy() 
    plant_knowledge = np.delete(plant_knowledge_np, 0, axis = 0)
                
    print(plant_knowledge)

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

"""
    #PRIORS 
    # For each informant's competence Di, I will be using an informative prior distribution 
    #   - Informative with the assumption that the informant will have some degree of prior knowledge
    #   about the subject based on cultural upbringing and assimilated knowledge  
    # For each consensus answer Zj, a beta prior will be used, as a result of the assumption 
    #   that the informant will have more prior knowledge than not on the subject. 
    #   - Using the Bernoulli distribution and a bernoulli prior assumping minimal prior 
    #       assumption. 

    N = len(data[:, 0])
    M = len(data[0])

    with pm.Model as model(): 

        #CHATGPT: After determining prior and distribution, used AI to help figure out 
        #   defining the priors given vector of size N and M
        D_b = pm.Beta('D_b', alpha = 6, beta = 4, shape = N)
        D = pm.binomial("D", M=M, b=b, observed=data) #Select priors 

        Z = pm.Bernoulli('Z', D_b, shape = M)

        #Defining the proability of p_ij. 
        D_reshaped = D[:, None] 
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped) #Is this for all i and j?

        #Define likelihood, linking the observed data X to the calculated proability 
        # p. 

#def data_sample(): 

"""



    

"""def run(): 
    plant_knowledge = open_dataset()
    py = pymc_model(plant_knowledge)
"""

if __name__ == '__main__': 
    plant_data = open_dataset()# dummy change
