import torch

train_file = ['data/Col-STAT/traindev_set.csv']
test_file = ['data/Col-STAT/test_set.csv']

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'

# real
# hyperparameters = dict(
#     model_name="Bert_9nodes_WO_Contrastiveloss",
#     NUM_EPOCHS=20,
#     MAX_SEQ_LENGTH=256, # 256 best, 128 for debug
#     GRADIENT_ACCUMULATION_STEPS=2,
#     WARMUP_STEPS=2,
#     NUM_LABELS=3,
# # Graph
#     n_steps=3,
#     n_nodes=9,
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
#     mid_dim = 512,
# # Model
#     hidden_dropout_prob=0.3 # default = 0.1
#     )

#debug
# train_file = ['data/Col-STAT/debug.csv']
# hyperparameters = dict(
#     #model_name="Bert_WO_Contrastiveloss",
#     model_name="debug",
#     NUM_EPOCHS=5,
#     MAX_SEQ_LENGTH=64, # 256 best, 128 for debug
#     GRADIENT_ACCUMULATION_STEPS=1,
#     WARMUP_STEPS=1,
#     NUM_LABELS=3,
#     Lambda = 0.5,
# # Graph
#     n_steps=1,
#     n_nodes=9,
#     max_length=64,
#     validation_split=0.1,
#     batch_size=1,
#     lr_bert=5e-6,
#     lr=5e-6,
#     lr_gamma=2,
#     lr_step=20,
#     clip_norm=50,
#     weight_decay=1e-4,
#     hidden_dim=768,
#     mid_dim=128,
# # # Model
#     hidden_dropout_prob=0.3 # default = 0.1
#     )


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
# #0 supernode that connected to everyone
mask[0][4], mask[0][3], mask[0][2], mask[0][1] = one, one, one, one
mask[0][5], mask[0][6], mask[0][7], mask[0][8], mask[0][9] = one, one, one, one, one
# #1 context node only connected to supernode
mask[1][0] = one
# #2: question node only connected to the supernode
mask[2][0], mask[2][1] = one, one
# #3, #4: 6 reference answer nodes that connect to supernode and question node #2
mask[3][0], mask[3][2] = one, one
mask[4][0], mask[4][2] = one, one
mask[5][0], mask[5][2] = one, one
mask[6][0], mask[6][2] = one, one
mask[7][0], mask[7][2] = one, one
mask[8][0], mask[8][2] = one, one
# #5: a student answer node that connect to supernode and #3, #4
mask[9][0], mask[9][3], mask[9][4] = one, one, one
mask[9][5], mask[9][6] = one, one
mask[9][7], mask[9][8] = one, one


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

q_rubric_dict = {'q2_a': ["We should use statistical inference because there could be cases of Carla just getting lucky if we just count how many times she gets a note right.",
                          "You could use statistical inference to determine if Carla has a good by comparing the distribution of her results to those of others.",
                          # p_correct
                          "I would use statistical inference to see if Carla has a good ear for music.",
                          "I don't think statistical inference should be used. What number of notes guessed  correctly would mean that she was above average? More than one person would need to be sampled for  this to make any sense.",
                          # In_correct
                          "No, there are no numerical values which could be applied to a Good ear for music",
                          "No, statistical inference should not be used in this scenario. Even if a student guesses all of the  notes played correctly, it does not mean he/she has a good ear for music "
                          ],
               'q2_b': ["There is a 1/7 chance that a student will guess the correct note. I would then take the ammount of times she guessed correctly (observed) compared to the number I expected her to guess (1/7). Once I have her observed count I can run a chi square test to determine if she was really guessing or has a good ear for music by looking at the p value.",
                        "We could run a number of note identification tests and then construct a 95% confidence interval stating the percentage of notes she gets right.",
                        # p_correct
                        "A random selection of students could guess 50 notes that were played. Then you could  calculate the mean, median, mode, Q1, Q3, etc. to see if the particular student was above average or not.",
                        "a series of trials would be ran and then mark if Carla is getting the correct or incorrect  after each guess.",
                        # In_correct
                        "ask a student to reproduce the tone and pitch of the note by himself/herself",
                        "The student is tested for a good ear for music by turning away from the piano as a teacher played  one of seven selected notes."
                        ],
               'q3_a': ["Personally, I would not use statistical inference in this scenario. Since the company must accept or reject the entire bulk, I would make sure that less than 5% of the screens are bad by testing all of them. If you rely on a statistic then there could sometimes be more and sometimes less than the 5% but I would not risk it in this case.",
                        "You should not use statistical inference for this because it would be easier to just count. If over 7 screens are bad, the whole lot can be rejected.",
                        #"no because statistical inference is the theory, methods, and practice of forming judgments about the parameters of a population, usually on the basis of random sampling",
                        # p_correct
                        #"In this case no, statistical inference should not be use because of the small percentage of  bad screens. If the percentage was greater than the 5% then it would be fine to use it. ",
                        #"No you shouldn't use statistical analysis ",
                        "Yes, a statistical inference should be used to figure out if the company should accept or  reject the full 150 piece order. If more than 5% of the screens are bad, they should reject them. So if more  than 7 and a half (7 or 8 depending on how you round) are bad, then they will send them back. ",
                        "You should just to get an accurate percentage to see if the shipment should be accepted  or rejected.",
                        # incorrect
                        "Yes because you can see what is the probability that the order is unuseable and decide whether  or not it’s a place to keep getting the screens from that supplier.",
                        "Yes because you can see what is the probability that the order is unuseable and decide whether  or not it’s a place to keep getting the screens from that supplier.",
                        ],
               'q3_b': ["If there are over 7 bad screens, they should reject.",
                        "count how many out of 150 are damaged then divide it by 150. if it is greater than .05 then reject it",
                        # p_correct
                        "The electronic companies should look at the percentage of bad screens from the  engineers. If the percentage is low then they should just send the screens back to the manufactures.  However if the percentage of bad screens is say greater than 25% then the company should reject the whole  bulk of screens.",
                        "have a set percentage of screens that are useable for you to accept if the percentage of  damaged screens are to high dont accept the screens. ",
                        # incorrect
                        "You could run a hypothesized test looking for the p-value of the bulk. In the test 5% or .05 would  be your alpha. And you would test at that significance to find the appropriate conclusion.",
                        "You could accept the order of display screens in the correlation is significant enough",
                        ],
               'q4_a': ["Student describes analysis of the average (or median) size/length/weight of fish caught byeach brother",
                        "Yes, statistical inference should be used in this scenario. We want to know if the averages for Mark and Dan differ too much to happen just by chance.",
                        # p_correct
                        "The statistics will reveal who gets a bigger fish on average and who gets more",
                        "Statistical inference should be used by comparing the two friends and which one is  catching the biggest fishes. ",
                        #"Yes, you could use statistical inference In this example. You could record the population  and the sample size. As well as do an observational analysis where you are comparing the two fishermen.  These are all statistical concepts.",
                        # incorrect
                        "No. Although there are ways to catch larger fish, there is also luck involved that may not come  from skill. ",
                        "No. Although there are ways to catch larger fish, there is also luck involved that may not come  from skill."
                        ],
               'q4_b': ["The mean fish lengths and variance of fish lengths for each fisherman's catch can be calculated. A two sample t-test can then be used to determine whether there exists a significant difference in mean lengths between the two samples. If there exists a significant difference in mean lengths, the better fisherman can be determined.",
                        "We would conduct a hypothesis test to see if Mark's average length is larger than Dan's.",
                        #p_correct
                        "Each brother would measure every fish they caught. They would come up with an  average length, and whoever had the largest average would be considered the better fisherman.",
                        " In order to determine which man is the better fisherman, the sample means of fish length  from each group would have to be calculated. After the sample means would be calculated, the researcher  could then look at other test statistics given through technology and compare these numbers. The man with  the larder sample mean and test statistics could be inferred to be the better fisherman of the two.",
                        # incorrect
                        "The better way to calculate who is the better fisherman would be to calculate the average  amount of fish caught as oppose to the average size",
                        " it might be roughly assumed from the day-to-day success evidence "
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
