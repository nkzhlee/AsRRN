'''
https://huggingface.co/models
Most used:

bert-base-uncased
distilbert-base-uncased
roberta-base
google/electra-small-discriminator
YituTech/conv-bert-base
'''
import torch

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
#debug
# TRAIN_FILE_PATH = ['../../data/Col-STAT/debug.csv']
# TEST_FILE_PATH = ['../../data/Col-STAT/debug_test_set.csv']


TRAIN_FILE_PATH = ['../../data/Col-STAT/traindev_set.csv']
TEST_FILE_PATH = ['../../data/Col-STAT/test_set.csv']
param = dict(
    train_id="0229_AsRRN_test_temp",
    model_name="/home/nlp/shared/models/huggingface/bert-base-uncased",
    #model_name = "roberta-base",
    random_seed=23,
    batch_size=1,
    lr=1e-5, # 1e-5
    epochs=23,
    lr_gamma=0.1,
    lr_step=15,
    max_length=128, 
    validation_split=0.2,
    weight_decay=0.0001,
    max_norm = 1.0, 
    WARMUP_STEPS=0.2, # 0.05
    GRADIENT_ACCUMULATION_STEPS=1,
    pre_step=5,
    #model
    hidden_dim=768,
    message_dim=128,
    hidden_dropout_prob=0.2,
    num_labels=3,
    #Graph
    n_steps=2,
    n_nodes=15,
    #contrastive learning
    Lambda=0.01,
    temp=1,
    norm=1,
    )

# wandb config
PROJECT_NAME = "YOUR_PROJECT"
ENTITY="YOUR ACCOUNT"
config_dictionary = dict(
    #yaml=my_yaml_file,
    params=param,
    )

# create message mask
n_nodes = param['n_nodes']
hidden_dim = param['hidden_dim']
message_dim = param['message_dim']
mask = torch.zeros(n_nodes+1, n_nodes+1, message_dim)
one = torch.ones(message_dim)
# #0 supernode that connected to everyone
mask[0][5], mask[0][4], mask[0][3], mask[0][2], mask[0][1] = one, one, one, one, one
mask[0][6], mask[0][7], mask[0][8], mask[0][9], mask[0][10]= one, one, one, one, one
mask[0][11], mask[0][12], mask[0][13], mask[0][14], mask[0][15]= one, one, one, one, one
# # p_random_node
# mask[0][10] = one
# #1 context node only connected to supernode
mask[1][0] = one
# #2: question node connected to the supernode and context node
mask[2][0], mask[2][1] = one, one
# #3, #4: 6 reference answer nodes that connect to supernode and question node #2
#correct
mask[3][0], mask[3][2] = one, one
mask[4][0], mask[4][2] = one, one
mask[5][0], mask[5][2] = one, one
mask[6][0], mask[6][2] = one, one
#pcorrect
mask[7][0], mask[7][2] = one, one
mask[8][0], mask[8][2] = one, one
mask[9][0], mask[9][2] = one, one
mask[10][0], mask[10][2] = one, one
#incorrect
mask[11][0], mask[11][2] = one, one
mask[12][0], mask[12][2] = one, one
mask[13][0], mask[13][2] = one, one
mask[14][0], mask[14][2] = one, one
# #5: a student answer node that connect to supernode and #3, #4
mask[15][0] = one
mask[15][3], mask[15][4], mask[15][5], mask[15][6] = one, one, one, one
mask[15][7], mask[15][8], mask[15][9], mask[15][10] = one, one, one, one
mask[15][11], mask[15][12], mask[15][13], mask[15][14] = one, one, one, one

q_context_dict = {'q2_a': "Some people who have a good ear for music can identify the notes they hear when music is played. One method of note identification consists of a music teacher choosing one of seven notes (A, B, C, D, E, F, G) at random and playing it on the piano. The student is asked to name which note was played while standing in the room facing away from the piano so that she cannot see which note the teacher plays on the piano. Should statistical inference be used to determine whether Carla has a “good ear for music”? Explain why you should or should not use statistical inference in this scenario." ,
               'q2_b': "Some people who have a good ear for music can identify the notes they hear when music is played. One method of note identification consists of a music teacher choosing one of seven notes (A, B, C, D, E, F, G) at random and playing it on the piano. The student is asked to name which note was played while standing in the room facing away from the piano so that she cannot see which note the teacher plays on the piano. explain how you would decide whether the student has a good ear for music using this method of note identification. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.",
               'q3_a': "An electronics company makes customized laptop computers for its customers by assembling various parts such as circuit boards, processors, and display screens purchased in bulk from other companies. The company regularly purchases bulk orders of 150 display screens from a supplier. If more than 5% of the display screens from the supplier are bad, the company may choose to reject the entire bulk order of 150 display screens for a refund. Otherwise the company must accept the entire bulk order of 150 display screens. Should statistical inference be used to determine whether the company should accept or reject the bulk order of display screens using the data gathered by the trained engineer? Explain why you should or should not use statistical inference in this scenario.",
               'q3_b': "An electronics company makes customized laptop computers for its customers by assembling various parts such as circuit boards, processors, and display screens purchased in bulk from other companies. The company regularly purchases bulk orders of 150 display screens from a supplier. If more than 5% of the display screens from the supplier are bad, the company may choose to reject the entire bulk order of 150 display screens for a refund. Otherwise the company must accept the entire bulk order of 150 display screens. explain how you would decide whether the electronics company should accept or reject the order of display screens using the data gathered by the trained engineer. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)",
               'q4_a': "Walleye is a popular type of freshwater fish native to Canada and the Northern United States. Walleye fishing takes much more than luck; better fishermen consistently catch larger fish using knowledge about proper bait, water currents, geographic features, feeding patterns of the fish, and more. Mark and his brother Dan went on a two ­week fishing trip together to determine who the better Walleye fisherman is. Each brother had his own boat and similar equipment so they could each fish in different locations and move freely throughout the area. They recorded the length of each fish that was caught during the trip, in order to find out which one of them catches larger Walleye on average. Should statistical inference be used to determine whether Mark or Dan is a better Walleye fisherman? Explain why statistical inference should or should not be used in this scenario.",
               'q4_b': "Walleye is a popular type of freshwater fish native to Canada and the Northern United States. Walleye fishing takes much more than luck; better fishermen consistently catch larger fish using knowledge about proper bait, water currents, geographic features, feeding patterns of the fish, and more. Mark and his brother Dan went on a two ­week fishing trip together to determine who the better Walleye fisherman is. Each brother had his own boat and similar equipment so they could each fish in different locations and move freely throughout the area. They recorded the length of each fish that was caught during the trip, in order to find out which one of them catches larger Walleye on average. explain how you would determine whether Mark or Dan is a better Walleye fisherman using the data from the fishing trip. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)"
}

q_text_dict = {'q2_a': "Should statistical inference be used to determine whether Carla has a “good ear for music”? Explain why you should or should not use statistical inference in this scenario." ,
               'q2_b': "explain how you would decide whether the student has a good ear for music using this method of note identification. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.",
               'q3_a': "Should statistical inference be used to determine whether the company should accept or reject the bulk order of display screens using the data gathered by the trained engineer? Explain why you should or should not use statistical inference in this scenario.",
               'q3_b': "explain how you would decide whether the electronics company should accept or reject the order of display screens using the data gathered by the trained engineer. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)",
               'q4_a': "Should statistical inference be used to determine whether Mark or Dan is a better Walleye fisherman? Explain why statistical inference should or should not be used in this scenario.",
               'q4_b': "explain how you would determine whether Mark or Dan is a better Walleye fisherman using the data from the fishing trip. (Be sure to give enough detail that a classmate could easily understand your approach, and how he or she would interpret the result in the context of the problem.)"
}
# correct reference answer
correct_ref_dict = {'q2_a': ["We should use statistical inference because there could be cases of Carla just getting lucky if we just count how many times she gets a note right.",
                          "Yes, you should use statistical inference. If Carla is just guessing randomly for the notes, we would expect her to guess right approximately 1/7th of the time in the long run. This would be the null hypothesis. From the discrepancy between the expected proportion of correct guesses and the real proportion of correct guesses we infer estimates about the quality of Carla's good ear for music.",
                          "Yes, because you would just have to choose a percentage that qualifies as a ""good ear for music"" and infer from the data if Carla has that.",
                          "I believe that statistical inference could be used to determine if Carla has an ear for music. I would look at the amount of times she was right compare to how many times she was wrong.  / If we are talking literally that she has an ear for music than yes this could be a statistical inference. However, I do not believe that it mean she is bad at music if she gets a lot wrong just that her ear to music isn't as good.",
                          ],
               'q2_b': ["There is a 1/7 chance that a student will guess the correct note. I would then take the ammount of times she guessed correctly (observed) compared to the number I expected her to guess (1/7). Once I have her observed count I can run a chi square test to determine if she was really guessing or has a good ear for music by looking at the p value.",
                        "We could run a number of note identification tests and then construct a 95% confidence interval stating the percentage of notes she gets right.",
                        "if the proportion of correctly identified notes was greater than the proportion of incorrectly identiftied notes beyond chance (significant p value)",
                        "Have the tester play a series of notes on the piano randomly, one at a time, after each one asking the student to make a guess as to which note was played. If the student after a large number of trials scored a significantly better score then average, maybe two standard deviations, then could be decided to have a good ear for music.",
                        ],
               'q3_a': ["Personally, I would not use statistical inference in this scenario. Since the company must accept or reject the entire bulk, I would make sure that less than 5% of the screens are bad by testing all of them. If you rely on a statistic then there could sometimes be more and sometimes less than the 5% but I would not risk it in this case.",
                        "You should not use statistical inference for this because it would be easier to just count. If over 7 screens are bad, the whole lot can be rejected.",
                        "no because statistical inference is the theory, methods, and practice of forming judgments about the parameters of a population, usually on the basis of random sampling",
                        #"No you should not use statistical inference, you should just keep record of how many screens are bad",
                        "Statistical inference is not necessary here. The engineer wants to know for each bulk whether more or less than 5%  of the screens are bad. Thus, he will investigate each bulk to draw his conclusion for that particular bulk, rather than testing hypothesis to make estimates for the population of bulk orders.",
                        ],
               'q3_b': ["If there are over 7 bad screens, they should reject.",
                        "count how many out of 150 are damaged then divide it by 150. if it is greater than .05 then reject it",
                        "If less than 5% of the screens are broken, the company should order the screens in bulk and save money.",
                        "Once the trained engineer has gathered his data, he would be able to determine the percentage of display screens from the supplier that are bad, if any at all. Anything less than 5% would not make too much of an impact on assembling, but if the amount of display screens that are bad comes out to be more than 5%, then the company should reject the whole order. Those are too many display screens that cannot be used for assembly. This would greatly affect work.",
                        ],
               'q4_a': ["Student describes analysis of the average (or median) size/length/weight of fish caught byeach brother",
                        "Yes, statistical inference should be used in this scenario. We want to know if the averages for Mark and Dan differ too much to happen just by chance.",
                        "Yes, statistical inference should be used to compare the different average values between the two groups.",
                        "Sure! Although the data might only be relevant to this particular outing as changes in fishing locations can have a great effect on the size outcomes of fish. We could examine mean or median fish length to determine who was the best walleye fisherman on this occasion.",
                        ],
               'q4_b': ["The mean fish lengths and variance of fish lengths for each fisherman's catch can be calculated. A two sample t-test can then be used to determine whether there exists a significant difference in mean lengths between the two samples. If there exists a significant difference in mean lengths, the better fisherman can be determined.",
                        "We would conduct a hypothesis test to see if Mark's average length is larger than Dan's.",
                        "null hypothesis: Mark's avg fish weight - Dan's avg fish weight = 0 / research hypothesis: Mark's avg fish weight - Dan's avg fish weight is NOT equal to 0 /  / Once it is determined that there is a difference between the two groups, then compare their average against each other /  / Mark's avg fish weight > Dan's avg fish weight / Mark's avg fish weight < Dan's avg fish weight",
                        "You would use a 2 sample t test to compare the means in the average length of fish",
                       ]
}
# partially correct reference answer
part_ref_dict = {'q2_a': [
                           "Yes. Since there is only one person being studied, we would have to infer that Carla represents the whole populatoin.",
                          "Yes, because there is a chance that she will not be able to identify the notes correctly, where she will then not have a good ear for music.",
                          "No because one person may think that she does have a good ear for music while someone else may disagree. This is more of an observation than something you can prove with statistics.",
                          "Statistical inference should not be used in this situation because a repetition of the piano method of note identification will increase her probability for accurately figuring out the notes over time. This would not accurately measure whether she has a good ear for music or not.",
                          ],
               'q2_b': ["I would have a predetermined list of notes that would be played and have a recorder stand with the correct answers and keep track of what Carla says and whether or not her answers were correct. After the data was collected I would calculate the percentage of time that Carla was able to correctly identify the music notes.",
                        "Play a series of notes, so that each note is played more than once.  Record what note was actually played and what note the student thought was played.  Compare how many they got right and how many they got wrong. if they got more right than wrong then they probably have a good ear for music.",
                        "Using this method of note identification it would be a simple process of many trials and recording if the student is correct and if the data shows the student is correct a large majority of the time (~75% +) the student then has a good ear for music.",
                        "If the student has more than 75% accuracy, they have a good ear for music.",
                        ],
               'q3_a': ["In this case no, statistical inference should not be use because of the small percentage of  bad screens. If the percentage was greater than the 5% then it would be fine to use it. ",
                        "No you shouldn't use statistical analysis ",
                        "Yes, a statistical inference should be used to figure out if the company should accept or  reject the full 150 piece order. If more than 5% of the screens are bad, they should reject them. So if more  than 7 and a half (7 or 8 depending on how you round) are bad, then they will send them back. ",
                        "You should just to get an accurate percentage to see if the shipment should be accepted  or rejected.",
                        ],
               'q3_b': ["The electronic companies should look at the percentage of bad screens from the  engineers. If the percentage is low then they should just send the screens back to the manufactures.  However if the percentage of bad screens is say greater than 25% then the company should reject the whole  bulk of screens.",
                        "have a set percentage of screens that are useable for you to accept if the percentage of  damaged screens are to high dont accept the screens. ",
                        "They should simply see if every screen works and if less than 5% work then you are in the clear.",
                        "Taking the information that we have, we can test the hypothesis that  5% of the display screens are bad.   / we would calculate the test statistic using the zp^ equation.  /  / zp^= p^-p/sqrt (p)(q)/n  /  / using this equation would give us the test statistic which we would find the z-value with to determine if the critical value falls within the rejection area to determine if in fact 15% or more of the screens will be defective and need to be rejected and customers refunded",
                        ],
               'q4_a': ["The statistics will reveal who gets a bigger fish on average and who gets more",
                        "Statistical inference should be used by comparing the two friends and which one is  catching the biggest fishes. ",
                        "Yes, statistical inference should be used to determine this because the data gathered by each person and averaged out is very accurate. Therefore you can easily see who the better fisherman is.",
                        "Yes, you could use statistical inference In this example. You could record the population  and the sample size. As well as do an observational analysis where you are comparing the two fishermen.  These are all statistical concepts.",
                        ],
               'q4_b': ["Each brother would measure every fish they caught. They would come up with an  average length, and whoever had the largest average would be considered the better fisherman.",
                        " In order to determine which man is the better fisherman, the sample means of fish length  from each group would have to be calculated. After the sample means would be calculated, the researcher  could then look at other test statistics given through technology and compare these numbers. The man with  the larder sample mean and test statistics could be inferred to be the better fisherman of the two.",
                        "If Mark averages a higher length of Walleye fish than Dan does, he is a better fisherman.",
                        "I would compare the average length of the fish caught by Mark and Dan. The brother with the larger average would be determined to be the better Walleye fisherman.",
                        ]
}
# incorrect reference answer
in_ref_dict = {'q2_a': [ "No, there are no numerical values which could be applied to a Good ear for music",
                          "I do not think that you should necessarily use a statistical inference because the student could potentially guess and get the answers correct",
                          "Statistical inference would not be applicable due to deciding whether Carla has a ""good ear for music"" is not reliant on herself. We would compare her ""good ear"" to that of others in a sample which would only imply that out of the group that is tested, Carla would be judged, instead of being tested upon her own competence",
                          "No, statistical inference should not be used in this scenario. Even if a student guesses all of the  notes played correctly, it does not mean he/she has a good ear for music "
                          ],
               'q2_b': ["ask a student to reproduce the tone and pitch of the note by himself/herself",
                        "The student is tested for a good ear for music by turning away from the piano as a teacher played  one of seven selected notes.",
                        "To test that I would play different notes at different distances to test how good and consistend her music hear is.",
                        "I would decide if a student has a good ear by choosing 3 songs to play. First song being the most recognizable song for a music savvy individual that would be easy to know the notes. Second, being slightly more challenging than the first, but simple enough to come up with at least a guess. Lastly, would be most challenging than first and second, and overall the test of evaluating a person's ear for music.",
                        ],
               'q3_a': ["Yes because you can see what is the probability that the order is unuseable and decide whether  or not it’s a place to keep getting the screens from that supplier.",
                        "Statistical inference should be used to analyze the data and figure out what percentage of the order is damaged",
                        "Yes, because you could need to determine if you reach the 5% of display screens.",
                        "Yes, statistical inference should be used to determine whether the company should accept or reject the bulk order of display screens using the data gathered by the trained engineer. I should use statistical inference because we to test whether the display screens can be accepted and if there is more than 5% of the screen is bad than we reject. We have to form a judgement on whether there are a significant amount of bad screens in the bulk in order to find out if we should accept it or reject it.",
                        ],
               'q3_b': ["You could run a hypothesized test looking for the p-value of the bulk. In the test 5% or .05 would  be your alpha. And you would test at that significance to find the appropriate conclusion.",
                        "You could accept the order of display screens in the correlation is significant enough",
                        "Using the 5% significance level, the computers will be determined to be accepted or rejected.",
                        "It would depend on the demand for the product. Would the company be able to recover from losing more days to having to return and later receive new screens? Or would they have less of a loss if they keep the few correct screens and continue to sell until new ones arrive.",
                       ],
               'q4_a': [ "No. Although there are ways to catch larger fish, there is also luck involved that may not come  from skill. ",
                        "no because we do not know exactly how different their tools may be",
                        "No, because they will be fishing in different areas one location might be more populated with Walleye fish than the other",
                        "Statistical inference should not be used to determine whether mark or dan is a better fisherman because there are too many outside forces which determine whether someone catches a fish or not. It would rather be suitable to use satistical inference to find the perfect spot at a specific point in time to fish but several observations would have to be nessecary in order to find sufficient evidence for that.",
                         ],
               'q4_b': ["The better way to calculate who is the better fisherman would be to calculate the average  amount of fish caught as oppose to the average size",
                        " it might be roughly assumed from the day-to-day success evidence ",
                        "No, you have to take in all the factors.",
                        "You cannot determine who is the better fisherman using the data from the fishing trip.",
                        ]
}
