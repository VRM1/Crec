'''Author: Vineeth Rakesh
This code is a part of our WSDM 2016 paper "Probabilistic Group recommendation Model for Crowdfunding Domains".
the paper uses kickstarter dataset; however, due to data privacy and other copy right issues, we are not
able to publish this dataset at this time. However, we have included a demo dataset from the famous movilens
data.'''

import numpy as np
from GetGroupData2 import fetch_grp_data
import json
import datetime
import os
import sys
import time
from django.utils.encoding import smart_str

dest_folder = '/home/dmkd172/CIKM15_grp_recc_codes/required_data2'
dest_folder2 = 'outputs'

class CrowdRecModel():
    
    def __init__(self,data,dat_typ,K,usr_id,itm_id,grp_sz,eta,alpha,rho):
        
        self.data = data
        self.K = K
        self.usr_ids = usr_id
        self.prj_ids = itm_id
        self.grp_sz = grp_sz
        self.dir = dest_folder2
        self.nusr = len(usr_id)
        self.nitm = len(itm_id)
        # prior variables
        self.eta = eta
        self.alpha = alpha
        self.rho = rho
        # declare main count variables of the model
        self.n_g_k = np.zeros((len(data),K))+alpha
        self.n_k_b = np.zeros((K,self.nusr))+eta
        self.n_b_v = np.zeros((self.nusr,self.nitm))
        ''' the variable n_v variable should be added with popularity priors
        . In this demo code popularity is just set uniform '''
        self.n_v = np.zeros((1,self.nitm))+0.01
        self.n_k_v = np.zeros((K,self.nitm))+0.01
        self.n_b_d = np.zeros((self.nusr,2))+rho
        # other count variables
        # a variable that keeps track of the number of topics across all user-itme pair in all groups
        self.n_k = np.zeros(K)
        # to keep track of which user-item pair has been assigned to which topic
        self.n_k_b_v = []
        # to keep track of which decision did a user choose i.e. 0 or 1
        self.n_d_b = []
        # scalars
        self.beta1 = 0.5
        self.beta2 = 0.5
        '''prior matrix for backer and project'''
        self.prior_n_b_v = np.zeros((self.nusr,self.nitm))
        if (dat_typ == 'movi'):
            # if it's movie lens dataset I keep an uniform prior
            self.n_b_v += 0.01
        else:
            self.create_usr_itm_prior()
            self.n_b_v = self.n_b_v + self.prior_n_b_v
        #initialize the initial count variables to some random counts
        for grp, itms in enumerate(self.data):
            n_k_b_v = []
            n_d_b =[]
            for usr_itm in itms:
                usr = usr_itm[0]
                itm = usr_itm[1]
                z = np.random.randint(0, K)
                d = np.random.randint(0, 2)
                self.n_g_k[grp,z] += 1
                self.n_k_b[z,usr] += 1
                self.n_v[0,itm] += 1
                self.n_b_d[usr,d] += 1
                # groups influence
                self.n_k_v[z,itm] += 1
                # his own influence
                self.n_b_v[usr,itm] += 1
                n_k_b_v.append(z)
                n_d_b.append(d)
                self.n_k[z] += 1
            self.n_d_b.append(np.array(n_d_b))
            self.n_k_b_v.append(np.array(n_k_b_v))
        
    '''method that creates prior matrix user backer and project. The priors are calculated based
    on our WSDM16 paper "probabilistic group recommendation model for crowdfunding domains". This 
    funciton does not apply for general movi lens dataset. Please note that the result of the
    recommendation model stronlgy relies on a properly formulated prior information about the 
    user-item.'''
    def create_usr_itm_prior(self):
        
        '''check if prior file already exist. Since creating prior is a tedious task
        I store these matrices based on the group size'''
        wrt_filnam = dest_folder+'/priors_4_grpsz'+str(self.grp_sz)+'.mat'
        try:
            matrix_file = open(wrt_filnam,'r')
            self.prior_n_b_v = np.load(matrix_file)
            print 'prior matrix loaded from file'
            return
        except:
            print 'prior matrix file does not exist going to create one...'
        # a dictionary that contains probability for all 1mi+ users 
        my_data_categ = json.loads(open(dest_folder+"/backer_category_probab_with_ids.json").read())
        my_data_creator = json.loads(open(dest_folder+"/backer_creator_probab_with_ids.json").read())
        my_data_prj_ids = json.loads(open(dest_folder+"/proj_ids_category_creator.json").read())
         
        # create prior for backer  and project
        count = 0

        for bkr in self.usr_ids:
            bkr_str = str(bkr)
            count += 1
            print 'done with %d backer'%(count)
            for proj in self.prj_ids:
                proj_str = str(proj)
                bkr_indx = self.usr_ids[bkr]
                prj_indx = self.prj_ids[proj]
                categ = my_data_prj_ids[proj_str]['category']
                creat = my_data_prj_ids[proj_str]['creator']
                # if both the category and creator are present in user's backing history
                if(categ in my_data_categ[bkr_str] and creat in my_data_creator[bkr_str]):
                    # get their probabilities
                    topic_prb = my_data_categ[bkr_str][categ]
                    creator_prb = my_data_creator[bkr_str][creat]          
                    total_prb =  topic_prb + creator_prb
                    # add this entry to the backer-project matrix
                    self.prior_n_b_v[bkr_indx,prj_indx] = total_prb
                # if just the topic category is present in user's backing history
                elif(categ in my_data_categ[bkr_str]):
                    topic_prb = my_data_categ[bkr_str][categ]
                    self.prior_n_b_v[bkr_indx,prj_indx] = topic_prb

                # if just the creator is present in user's backing history
                elif(creat in my_data_creator[bkr_str]):
                    creator_prb = my_data_creator[bkr_str][creat]
                    self.prior_n_b_v[bkr_indx,prj_indx] = creator_prb
                # none of the topic or the creator is present in user's backing history
                else:
                    continue
        np.save(wrt_filnam,self.prior_n_b_v)
        row_len = self.prior_n_b_v.shape[0]
        clm_len = self.prior_n_b_v.shape[1]
        tot_non_zero = len(np.transpose(np.nonzero(self.prior_n_b_v)))
        density = (tot_non_zero)/float(row_len*clm_len)
        print '1) created the prior matrix of density %2f for backer-project......'%(density)
        
    # creating project specific priors that includes popularity and reward status 
    def create_prj_prior(self):
        
        # the popularity scores of project at 3% 50% and 80% of their total duration
        pop_proj = json.loads(open("data/project_popularity_prior.json").read())
        count = 0
        for proj in self.prj_ids:
            prj_indx = self.prj_ids[proj]
            if(proj in pop_proj):
                # fetching the popularity on 50% of the project duration
                pop_score = np.round(pop_proj[proj][1]/float(100),2)
                if(pop_score > 1):
                    pop_score = 1
                self.n_v[0,prj_indx] = pop_score
        print '1) created prior matrix for project based on popularity..creating priors of users...'
    
    def neg_test(self, n_g_k,n_k_b,n_b_d,n_k_v):
        
        if(n_k_v < 0 ):
            print '-ve found'
            sys.exit(1)
            
        
            
    def CrowdRec_learning(self, no_of_itr):
        
        tot_grps = len(self.data)
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        itr_number = open('tmp/itr_num_gp'+str(self.grp_sz),'w')
        for itr in range(no_of_itr):
            itr_number.write(str(itr)+'\n')
            itr_number.flush()
            
            for grp, itms in enumerate(self.data):
                
                grps_remain = tot_grps-grp+1
                new_n_k_b_v = []
                new_n_d_b = []
                n_g_k = self.n_g_k[grp]
                n_k_b_v = self.n_k_b_v[grp]
                n_d_b = self.n_d_b[grp]
                for indx, usr_itm in enumerate(itms):
                    usr = usr_itm[0]
                    itm = usr_itm[1]
                    # obtain the assigned topic
                    z = n_k_b_v[indx]
                    d = n_d_b[indx]
                    # decrement the topic counts
                    n_g_k[z] -= 1
                    self.n_k_b[z,usr] -= 1
                    self.n_v[0,itm] -= 1
                    self.n_b_d[usr,d] -= 1
                    if(d == 0):
                        # the user moves with group's decision
                        self.n_k_v[z,itm] -= 1
                    else:
                        # user goes with his own decision
                        self.n_b_v[usr,itm] -= 1
                    self.n_k[z] -= 1
                    self.neg_test(n_g_k[z],self.n_k_b[z,usr],self.n_b_d[usr,d],self.n_k_v[z,itm])
                     
                    # Sample the decision variable
                    # if the user choses topic based on groups preference
                    p_z_0 = np.divide(n_g_k, n_g_k.sum()) * np.divide(self.n_k_b[:,usr],self.n_k) * \
                            (np.divide(self.n_k_v[:,itm],self.n_k) + self.beta1 * np.divide(self.n_v[0,itm], self.n_v.sum()))
#                     p_z_0 = n_g_k * np.divide(self.n_k_b[:,usr],self.n_k) * np.divide(self.n_k_v[:,itm],self.n_k)
                    # from some reason p_z_0 exceeds 1 at times. i have to investiage why. to solve this i normalize p_z_0
                    if(p_z_0.sum() < 1):
                        to_add = (1-p_z_0.sum())/float(len(p_z_0))
                        p_z_0 += to_add
                    if(p_z_0.sum() > 1):
                        p_z_0 = np.divide(p_z_0,p_z_0.sum())
                    
                    try:
                        new_pz0 = np.random.multinomial(1, p_z_0).argmax()
                    except:
                        print self.n_k_v[z,itm]
                        sys.exit(1)

                        
                    # if user chooses topic based on his own preference
                    p_z_1 = np.divide(n_g_k, n_g_k.sum()) * np.divide(self.n_k_b[:,usr],self.n_k)
#                     p_z_1 = n_g_k * np.divide(self.n_k_b[:,usr],self.n_k)
                    if(p_z_1.sum() < 1):
                        to_add = (1-p_z_1.sum())/float(len(p_z_1))
                        p_z_1 += to_add
                        
                    if(p_z_1.sum() > 1):
                        p_z_1 = np.divide(p_z_1,p_z_1.sum())
                    
                    try:
                        new_pz1 = np.random.multinomial(1, p_z_1).argmax()
                    except:
                        p_z_1 = np.divide(p_z_1,p_z_1.sum())
                        new_pz1 = np.random.multinomial(1, p_z_1).argmax()
                        
                             
                    p_d_0 = np.divide(self.n_b_d[usr,0], (self.n_b_d[usr,0] + self.n_b_d[usr,1])) * \
                            (np.divide(self.n_k_v[z,itm], self.n_k[z]) + self.beta1 * np.divide(self.n_v[0,itm], self.n_v.sum()))
                             
                    p_d_1 = np.divide(self.n_b_d[usr,1], (self.n_b_d[usr,0] + self.n_b_d[usr,1])) * \
                            (np.divide(self.n_b_v[usr,itm], self.n_b_v[usr].sum()) + self.beta2 * np.divide(self.n_v[0,itm], self.n_v.sum())) 
                     
                    # the final decision by user
                    usr_dec = np.array([p_d_0,p_d_1]).argmax()
                    # update decision
                    self.n_b_d[usr,usr_dec] += 1
                    if(usr_dec == 0):
                        # update group-topic
                        n_g_k[new_pz0] += 1
                        # update topic-backer
                        self.n_k_b[new_pz0,usr] += 1
                        #update topic-item
                        self.n_k_v[new_pz0,itm] += 1
                        # update the popularity count
                        self.n_v[0,itm] += 1
                        # reinitialize the topic for user-item pair since new topic has been chosen
                        new_n_k_b_v.append(new_pz0)
                        # reinitialize the new decision that the user has chosen
                        new_n_d_b.append(usr_dec)
                        self.n_k[new_pz0] += 1
                    else:
                        # user chooses project according to his own preference
                        n_g_k[new_pz1] += 1
                        self.n_k_b[new_pz1,usr] += 1
                        self.n_b_v[usr,itm] += 1
                        self.n_v[0,itm] += 1
                        new_n_k_b_v.append(new_pz1)
                        new_n_d_b.append(usr_dec)
                        self.n_k[new_pz1] += 1

                print 'done with group %d which had %d user-itm pairs, %d groups still remain'%(grp,len(itms),grps_remain)
            
                ''' update the new topic and decision assignments for user-item pair
                for each group by removing the old assignments'''
             
                self.n_k_b_v[grp] = new_n_k_b_v
                self.n_d_b[grp] = new_n_d_b
        
        print '*****learning complete writing the matrix to files...'
        self.__write_matrices()
    # method that writes the final matrix to file
    def __write_matrices(self):
        
        
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        grp_tp_mat = open(self.dir+'/group_topic.mat','w')
        tp_bkr_mat = open(self.dir+'/topic_bkr.mat','w')
        bkr_dec_mat = open(self.dir+'/bkr_decision.mat','w')
        tp_prj_mat = open(self.dir+'/topic_project.mat','w')
        bkr_prj_mat = open(self.dir+'/bkr_project.mat','w')
        '''we create one more variable for list of user-item pairs for each group
        this json file should serve as the matrix indices 
        when evaluating the recommendation'''
        grp_bkings = dict()
        np.save(grp_tp_mat,self.n_g_k)
        np.save(tp_bkr_mat,self.n_k_b)
        np.save(bkr_dec_mat,self.n_b_d)
        np.save(tp_prj_mat,self.n_k_v)
        np.save(bkr_prj_mat,self.n_b_v)
        for grp,usr_itm in enumerate(self.data):
            grp_bkings[grp] = usr_itm
        with open(self.dir+'/data_4_gpsz'+str(self.grp_sz)+'.json','w') as fp:
            json.dump(grp_bkings,fp)
        print 'written the output matrix and necessary files to %s folder'%(self.dir)
        

def normalize_mat(matrixdata,filename,option='row'):
        if (option == 'row'):
            rowsum = matrixdata.sum(axis=1)
            rowsum = rowsum[:,np.newaxis]
            matrixdata = np.divide(matrixdata,rowsum)
            with open(filename,'w') as fp:
                np.save(fp,matrixdata)
            
        else:
            clmsum = matrixdata.sum(axis=1)
            matrixdata = np.divide(matrixdata,clmsum)

# this normalizes the matrix according to the parameter calculation in the paper
def normalizeLearntData(grp_sz,topics):
        
    folder = dest_folder2
    if (os.path.isdir(folder)):
        grp_tp_mat = np.load(folder+'/group_topic.mat')
        grp_tp_mat = normalize_mat(matrixdata=grp_tp_mat,filename=folder+'/group_topic_norm.mat', option='row')
        tp_bkr_mat = np.load(folder+'/topic_bkr.mat')
        tp_bkr_mat = normalize_mat(matrixdata=tp_bkr_mat,filename=folder+'/topic_bkr_norm.mat', option='row')
        bkr_dec_mat = np.load(folder+'/bkr_decision.mat')
        bkr_dec_mat = normalize_mat(matrixdata=bkr_dec_mat,filename=folder+'/bkr_decision_norm.mat', option='row')
        tp_prj_mat = np.load(folder+'/topic_project.mat')
        tp_prj_mat = normalize_mat(matrixdata=tp_prj_mat,filename=folder+'/topic_project_norm.mat',option='row')
        bkr_prj_mat = np.load(folder+'/bkr_project.mat')
        bkr_prj_mat = normalize_mat(matrixdata=bkr_prj_mat,filename=folder+'/bkr_project_norm.mat', option='row')
        print 'normalized all matrices'
    else:
        print '%s folder is missing cannot perform operation'%(folder)
        

# method that calculates the recommendation score of group to project using the learnt set of matrices
def get_recc_matrix(grp_sz=5,itm_sz=5,dat_typ='kck',topics=25):
    if (dat_typ=='movi'):
        folder=dest_folder2
   
    if (os.path.isdir(folder)):
        data = []
        itm_list = set()
        
        rd_fil = open('movlens/pcc_filteredgroup_sz'+str(grp_sz)+'itm'+str(itm_sz)+'.csv','r')
        for line in rd_fil:
            line = line.strip()
            rd_line = line.split('\t')
            data.append([])
            index = len(data)-1
            for val in rd_line[1:]:
                val = val.split(',')
                usr_itm = (int(val[0]),int(val[1]))
                itm_list.add(int(val[1]))
                data[index].append(usr_itm)
            
        grp_tp_mat = np.load(folder+'/group_topic_norm.mat')
        tp_bkr_mat = np.load(folder+'/topic_bkr_norm.mat')
        bkr_dec_mat = np.load(folder+'/bkr_decision_norm.mat')
        tp_prj_mat = np.load(folder+'/topic_project_norm.mat')
        bkr_prj_mat = np.load(folder+'/bkr_project_norm.mat')
        print 'loaded all matrices'
        
        tot_itms = bkr_prj_mat.shape[1]
        tot_grps = len(data)
        grp_prj_recc = np.zeros((tot_itms,tot_grps))
        print 'total items %d, total groups %d...creating the preference matrix'%(tot_itms,tot_grps)
        count = 0
        for itm in itm_list:
            count += 1
            grp_cnt = 0
            for grp,vals in enumerate(data):  
                grp_cnt += 1
                # get the topic array of the group
                gtopic = grp_tp_mat[grp]
                val  = data[grp]
                prefscore = 0
                for user_itm in vals:
                    usr = user_itm[0]
#                     itm = user_itm[1]
                    # get the topic array of the backer
                    tpbkr = tp_bkr_mat[:,usr]
                    # decision variable if backers chooses grou's preference
                    d_0 = bkr_dec_mat[usr,0]
                    # decision variable if backers chooses his own preference
                    d_1 = bkr_dec_mat[usr,1]
                    # select the bker-itm cell from backer itm matrix
                    bkr_itm_choice = bkr_prj_mat[usr,itm]
                    # select the topic array for the project
                    tpprj = tp_prj_mat[:,itm]
                    # prefernce score for an item for the user
                    prefscore += (gtopic * tpbkr * (d_0 * tpprj + d_1 * bkr_itm_choice)).sum()
                # the agregated score of every individual becomes a group's total preference score
                grp_prj_recc[itm,grp] = prefscore
                print 'for item %d done with group %d'%(count,grp_cnt)
                # although there are 400000 groups get the final recommendation for so many groups will take 350hrs!
            print 'item %d done'%(count)    
        with open(folder+'/ComFinalReccScore_gpsz'+str(grp_sz)+'.mat','w') as fp:
            np.save(fp,grp_prj_recc)
    else:
        print '%s folder is missing cannot perform operation'%(folder)
                

def Main(dat_typ,grp_sz,topics):
    
#    topics = 25
    gp_sz = grp_sz
    itr_cnt = 10
    alpha = np.round(50/float(250),1)
    (data,usr_id,itm_id) = fetch_grp_data(grp_sz=grp_sz,itm_sz=itm_sz,pcc_cutoff=0.2,data_nam=dat_typ)
    t_grps = len(data)
    t_usrs = len(usr_id)
    t_itmid = len(itm_id)
    now = datetime.datetime.now()
    print now
    print '*******starting crowdrec model for a grp sz:%d,uniq users:%d, uniq items:%d, initializing all variables...'%(t_grps,t_usrs,t_itmid)
    time.sleep(10)
    ob_crowd_rec = CrowdRecModel(data=data,dat_typ=dat_typ,K=topics,usr_id=usr_id,itm_id=itm_id,grp_sz=gp_sz,eta=0.01,alpha=alpha,rho=0.5)
    print '2) finished initializing all the matrices starting the learning process....'
    ob_crowd_rec.CrowdRec_learning(itr_cnt)
    now = datetime.datetime.now()
    print now
    return topics

if __name__ == '__main__':
    
    grp_sz = 5
    itm_sz = 5
    dat_typ='movi'
    tpc=[100]
    for val in tpc:
        topics=val
        Main(dat_typ,grp_sz,topics)
        normalizeLearntData(grp_sz,topics)
        get_recc_matrix(grp_sz=grp_sz,itm_sz=itm_sz,dat_typ=dat_typ,topics=topics)
