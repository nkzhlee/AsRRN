import torch



train_file = ['data/Col-STAT/traindev_set.csv']
test_file = ['data/Col-STAT/test_set.csv']


SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'

# real
# hyperparameters = dict(
#     model_name="Bert_WO_Contrastiveloss",
#     NUM_EPOCHS=1,
#     MAX_SEQ_LENGTH=256, # 256 best, 128 for debug
#     GRADIENT_ACCUMULATION_STEPS=2,
#     WARMUP_STEPS=2,
#     NUM_LABELS=3,
# # Graph
#     n_steps=2,
#     n_nodes=4,
#     max_length=256,
#     validation_split = 0.1,
#     batch_size = 1,
#     lr_bert = 5e-6,
#     lr = 5e-6,
#     lr_gamma = 2,
#     lr_step = 20,
#     clip_norm = 50,
#     weight_decay = 1e-4,
#     hidden_dim = 768,
#     mid_dim = 512
#     )

# debug
hyperparameters = dict(
    #model_name="Bert_WO_Contrastiveloss",
    model_name="debug",
    NUM_EPOCHS=1,
    MAX_SEQ_LENGTH=128, # 256 best, 128 for debug
    GRADIENT_ACCUMULATION_STEPS=1,
    WARMUP_STEPS=1,
    NUM_LABELS=3,
# Graph
    n_steps=2,
    n_nodes=4,
    max_length=128,
    validation_split = 0.1,
    batch_size = 1,
    lr_bert = 5e-6,
    lr = 5e-6,
    lr_gamma = 2,
    lr_step = 20,
    clip_norm = 50,
    weight_decay = 1e-4,
    hidden_dim = 768,
    mid_dim = 256
    )


# wandb config
config_dictionary = dict(
    #yaml=my_yaml_file,
    params=hyperparameters,
    )


# create message mask
n_nodes = hyperparameters['n_nodes']
hidden_dim = hyperparameters['hidden_dim']
mid_dim = hyperparameters['mid_dim']
mask = torch.zeros(n_nodes+1, n_nodes+1, mid_dim)
one = torch.ones(mid_dim)
# #0: question node only connected to the supernode
mask[0][4] = one
# #1, #2: two reference answer nodes that connect to supernode and #0
mask[1][4], mask[1][0] = one, one
mask[2][4], mask[2][0] = one, one
# #3: a student answer node that connect to supernode and #2, #3
mask[3][4], mask[3][2], mask[3][1] = one, one, one
# #4: supernode that connected to everyone
mask[4][3], mask[4][2], mask[4][1], mask[4][1] = one, one, one, one


q_text_dict = {'q2_a': "Should statistical inference be used to determine whether Carla has a “good ear for music”? Explain why you should or should not use statistical inference in this scenario." ,
               'q2_b': "explain how you would decide whether the student has a good ear for music using this method of note identification. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.",
               'q3_a': "Should statistical inference be used to determine whether the company should accept or reject the bulk order of display screens using the data gathered by the trained engineer? Explain why you should or should not use statistical inference in this scenario.",
               'q3_b': "explain how you would decide whether the electronics company should accept or reject the order of display screens using the data gathered by the trained engineer. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)",
               'q4_a': "Should statistical inference be used to determine whether Mark or Dan is a better Walleye fisherman? Explain why statistical inference should or should not be used in this scenario.",
               'q4_b': "explain how you would determine whether Mark or Dan is a better Walleye fisherman using the data from the fishing trip. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)"
}

q_context_dict = {'q2_a': "Some people who have a good ear for music can identify the notes they hear when music is played. One method of note identification consists of a music teacher choosing one of seven notes (A, B, C, D, E, F, G) at random and playing it on the piano. The student is asked to name which note was played while standing in the room facing away from the piano so that she cannot see which note the teacher plays on the piano. Should statistical inference be used to determine whether Carla has a “good ear for music”? Explain why you should or should not use statistical inference in this scenario." ,
               'q2_b': "Some people who have a good ear for music can identify the notes they hear when music is played. One method of note identification consists of a music teacher choosing one of seven notes (A, B, C, D, E, F, G) at random and playing it on the piano. The student is asked to name which note was played while standing in the room facing away from the piano so that she cannot see which note the teacher plays on the piano. explain how you would decide whether the student has a good ear for music using this method of note identification. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.",
               'q3_a': "An electronics company makes customized laptop computers for its customers by assembling various parts such as circuit boards, processors, and display screens purchased in bulk from other companies. The company regularly purchases bulk orders of 150 display screens from a supplier. If more than 5% of the display screens from the supplier are bad, the company may choose to reject the entire bulk order of 150 display screens for a refund. Otherwise the company must accept the entire bulk order of 150 display screens. Should statistical inference be used to determine whether the company should accept or reject the bulk order of display screens using the data gathered by the trained engineer? Explain why you should or should not use statistical inference in this scenario.",
               'q3_b': "An electronics company makes customized laptop computers for its customers by assembling various parts such as circuit boards, processors, and display screens purchased in bulk from other companies. The company regularly purchases bulk orders of 150 display screens from a supplier. If more than 5% of the display screens from the supplier are bad, the company may choose to reject the entire bulk order of 150 display screens for a refund. Otherwise the company must accept the entire bulk order of 150 display screens. explain how you would decide whether the electronics company should accept or reject the order of display screens using the data gathered by the trained engineer. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)",
               'q4_a': "Walleye is a popular type of freshwater fish native to Canada and the Northern United States. Walleye fishing takes much more than luck; better fishermen consistently catch larger fish using knowledge about proper bait, water currents, geographic features, feeding patterns of the fish, and more. Mark and his brother Dan went on a two ­week fishing trip together to determine who the better Walleye fisherman is. Each brother had his own boat and similar equipment so they could each fish in different locations and move freely throughout the area. They recorded the length of each fish that was caught during the trip, in order to find out which one of them catches larger Walleye on average. Should statistical inference be used to determine whether Mark or Dan is a better Walleye fisherman? Explain why statistical inference should or should not be used in this scenario.",
               'q4_b': "Walleye is a popular type of freshwater fish native to Canada and the Northern United States. Walleye fishing takes much more than luck; better fishermen consistently catch larger fish using knowledge about proper bait, water currents, geographic features, feeding patterns of the fish, and more. Mark and his brother Dan went on a two ­week fishing trip together to determine who the better Walleye fisherman is. Each brother had his own boat and similar equipment so they could each fish in different locations and move freely throughout the area. They recorded the length of each fish that was caught during the trip, in order to find out which one of them catches larger Walleye on average. explain how you would determine whether Mark or Dan is a better Walleye fisherman using the data from the fishing trip. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)"
}

q_rubric_dict = {'q2_a': ["Student provides rationale with accommodation for variability (e.g. repeat test method many times; compare to chance model)", "Student describes analysis of probability/proportion/number of correct-incorrect", "Student advocates use of statistical inference",
                          "We should use statistical inference because there could be cases of Carla just getting lucky if we just count how many times she gets a note right.",
                          "Student advocates use of statistical inference AND Student provides rationale with accommodation for variability (e.g. repeat test method many times)",
                          "It is sufficient that the student only implies repeating the test method many times (i.e.“count how many times she gets a note right” in this case)",
                          "You could use statistical inference to determine if Carla has a good by comparing the distribution of her results to those of others.",
                          "Student advocates use of statistical inference AND Student provides rationale with accommodation for variability (e.g. repeat test method many times)",
                          "It is sufficient that the student only implies repeating the test method many times (i.e. “distribution of her results” in this case)"
                          ],
               'q2_b': ["Student names OR paraphrases a binomial procedure, one-proportion Z-procedure, or analogous inferential method (e.g. inferential analysis of probability/proportion/number of correct-incorrect)",
                        "There is a 1/7 chance that a student will guess the correct note. I would then take the ammount of times she guessed correctly (observed) compared to the number I expected her to guess (1/7). Once I have her observed count I can run a chi square test to determine if she was really guessing or has a good ear for music by looking at the p value.",
                        "We could run a number of note identification tests and then construct a 95% confidence interval stating the percentage of notes she gets right."
                        ],
               'q3_a': ["Student does NOT advocate for statistical inference (i.e. “No”), AND recognizes that the engineer/company has access to complete information about the order.",
                        "Student describes that decision is made directly from the number/proportion of bad displays",
                        "Student describes that engineer has access to entire population of interest",
                        "Student describes that there is no sampling",
                        "Personally, I would not use statistical inference in this scenario. Since the company must accept or reject the entire bulk, I would make sure that less than 5% of the screens are bad by testing all of them. If you rely on a statistic then there could sometimes be more and sometimes less than the 5% but I would not risk it in this case.",
                        "You should not use statistical inference for this because it would be easier to just count. If over 7 screens are bad, the whole lot can be rejected.",
                        "no because statistical inference is the theory, methods, and practice of forming judgments about the parameters of a population, usually on the basis of random sampling"
                        ],
               'q3_b': ["Student recommends using the absolute threshold based on engineering data (5% OR 7.5 screens; 7 or 8 are both accepted)",
                        "If there are over 7 bad screens, they should reject.",
                        "count how many out of 150 are damaged then divide it by 150. if it is greater than .05 then reject it"
                        ],
               'q4_a': ["Student advocates use of statistical inference",
                        "Student provides rationale with accommodation for sampling variability (e.g. compare to chance model)",
                        "Student describes analysis of the average (or median) size/length/weight of fish caught byeach brother",
                        "Yes, statistical inference should be used in this scenario. We want to know if the averages for Mark and Dan differ too much to happen just by chance."
                        ],
               'q4_b': ["Student names appropriate inferential statistical method (e.g. two-sample t-procedure, oranalogous)",
                        "Student describes inferential statistical analysis of the average (e.g. mean, median) size/length/weight of fish caught by each brother",
                        "The mean fish lengths and variance of fish lengths for each fisherman's catch can be calculated. A two sample t-test can then be used to determine whether there exists a significant difference in mean lengths between the two samples. If there exists a significant difference in mean lengths, the better fisherman can be determined.",
                        "We would conduct a hypothesis test to see if Mark's average length is larger than Dan's."
                        ]
}


# get the answer stat
from pytorch_transformers import BertTokenizer
tokenizer_2 = BertTokenizer.from_pretrained('bert-base-uncased')
num_token = 0
print("NUM of Question", len(q_text_dict))
for q in q_text_dict.values():
    tokens = tokenizer_2.tokenize(q)
    num_token += len(tokens)
print("The ave length of question is ", str(num_token/len(q_text_dict)))
