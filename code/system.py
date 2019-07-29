"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
import difflib
import re

def divergence(class1, class2):
    """compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12

def reduce_dimensions(feature_vectors_full, model):
    """Method that reduces the dimensions of the data using the chosen features.

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    features
    """
    #the chsoen features from the training stage
    features = model['features']
    
    #Sets variable for the principal components of the train data
    V = np.array(model['Principal_Components'])

    #projecting data onto principal component axes
    pcatrain_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), V)   
    
    return pcatrain_data[:, features]


def choose_features(feature_vectors_full, model):
    """Method uses divergence on the PCA features to choose the 10 best PCA components.
    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    features
    """
    
    train_labels = np.array(model['labels_train'])
    V = np.array(model['Principal_Components'])
    
    #projecting data onto principal component axes
    pcatrain_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), V)
    
    #characters used for pairwise divergence analysis, notice low frequency character
    #such as q,z,j have been omitted
    #characters = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","r","s","t","u","v","w","y","x",".",",","?","!"]
    characters = ["a","b","c","d","e","f","g","h","i","k","l","m","n","o","p","r","s","t","u","v","w","y","x","?",".","!",","]
    #array that stores the cumulative divergence totals of the PCA's
    cumulative_divergence_totals=[]
    
    #divergence sum for letter pairs e.g. (a,b), (a,c), (a,d)
    letter_pair_sum=0
    #divergence sum of all letter pairs for a PCA component
    PCA_sum=0
    
    #loops through each pca feature and calculate its 1d divergence
    #NOTE it calculates the cumulative 1d divergence e.g. the 27th PCA is the divergence of
    #the 27th PCA and the 26 before it
    for f in range(0, 40):
        PCA_sum=0    
        for i in range(0, 26):
            #selects next character e.g. 1st chaacter a
            char1=characters[i]
            letter_pair_sum=0
            for j in range(0, 26):  
                
                #gets all characters leading from character 1 and does individul pair-wise analysis
                char2=characters[j+1]
                class1 = pcatrain_data[train_labels[:] ==char1,:f]
                class2 = pcatrain_data[train_labels[:] ==char2,:f]
                #uses the classes of the letters and calcualtes their divergence
                d12 = divergence(class1, class2) 
                #cumulative divergence sum for each letter comapred with other letters
                letter_pair_sum=sum(d12)+letter_pair_sum
                
            PCA_sum=PCA_sum+letter_pair_sum
        cumulative_divergence_totals.append(PCA_sum)
    
    #Used to get the individual 1-d divergence of each feature using the cumulative 1d PCA divergence totals
    divergence_feature=[]
    for i in range(0, 40):
        if i > 0:
            value = cumulative_divergence_totals[i]-cumulative_divergence_totals[i-1]
            divergence_feature.append(value)
        else:
            divergence_feature.append(cumulative_divergence_totals[i])
            
    #gets the index positions of the top 10 individual PCA divergence values
    ten_highest_index = np.array(divergence_feature)
    ten_highest_index = np.argpartition(ten_highest_index, -10)[-10:].tolist()
    
    #uses the index posiitons of the top 10 PCA divergence values as the chosen features
    features=ten_highest_index
    return features
    

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors

# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    #Create a dictionary to store and return results of training stage
    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    
    #Here I compute the eigenvectovs of the covariance matrix using the training data,
    #to compute the first 40 principal components
    covx = np.cov(fvectors_train_full, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
    v = np.fliplr(v)
    
    #I then store the principal comonents "V"
    model_data['Principal_Components'] = v.tolist()
    
    #Gets a lsit of the ten chosen features and stores them in the dicitonarys 
    model_data['features']=choose_features(fvectors_train_full, model_data)
    print(model_data['features'])
    
    #Performs the dimentsionality reduiction of the training data
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)
    
    #Stores the training data after its dimensions have been reduced
    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    
    # Perform the dimensionality reduction of the test data
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced

def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:
    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    
    #train data and labels are set using data from model dictionary
    train = np.array(model['fvectors_train'])
    train_labels = np.array(model['labels_train'])

    #the test data
    test=page
    
    #Implementation of nearest neighbour classification
    x= np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # using cosine distance
    nearest=np.argmax(dist, axis=1)
    label = train_labels[nearest]
    
    #returns the labels of the classified test data
    return label

def correct_errors(page, labels, bboxes, model):
    """error correction. Returns labels unchanged as could not
     consistently identif bounding boxes of spaces between words, 
     code of what i tried is below, see error correction justification 
     in report.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    
    pagelength=len(bboxes)
    
    pagelength=pagelength-100
    #file containing english word list
    File = open("wordsEn.txt") 
    
    #The output of the test labels after adding spaces based on bboxes
    output_string="" 
    #output of each word identified using bboxes
    output_word=""
    #word used to find a match in the wordsEn.txt
    finalword=""
    #matching word if found
    close_Match=""
    #counter used to count the nubmer of letters in a word
    count=0
    
    #goes through the length of the test data and compares the bounding box between a character 
    #its next successive character
    for i in range(0,pagelength):
        letter=bboxes[i]
        nextletter=bboxes[i+1]
        #calcualtes the difference between the two characters a and c co-ordinates
        adiff=nextletter[0]-letter[0]
        cdiff=nextletter[2]-letter[2]
        
        #if the difference in the a and c co-ordinates is above a threshold a space is
        #inserted in the output string and a word has been identified, then gets the labels
        #classified for that word
        if ((adiff>20) and (cdiff >23)) or ((adiff>23) and (cdiff >20)):
            output_string=output_string+labels[i]+" "
            finalword=output_word
            output_word=""
            #checks if the classifed word exists
            if CheckwordExists(finalword):
                #if does not exist finds closest matching word
                close_Match=difflib.get_close_matches(finalword, File,1,0.2)
                #replaces labels of classifed word with closest matching word
                if close_Match :
                    finalword=close_Match[0]
                    #COMMENTED OUT BECAUSE NOT CONSISTENT, SEE REPORT
                    #labels[i-count:count]=finalword        
            count=0       
        else:
            output_string=output_string+labels[i]
            output_word=output_word+labels[i]
            count=count+1
            
    #returns the labels of the classified test data     
    return labels

#fucntion used to check if a given word matches an exact word in the wordsEn.txt file
#which contains a list of all valid english words
def CheckwordExists(word):
    with open("wordsEn.txt", 'r') as file:
        if re.search('^{0}$'.format(re.escape(word)), file.read(), flags=re.M):
            return False
        else:
            return True
