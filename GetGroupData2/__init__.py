'''Author: Vineeth Rakesh
This code is a part of our WSDM 2016 paper "Probabilistic Group recommendation Model for Crowdfunding Domains".
the paper uses kickstarter dataset; however, due to data privacy and other copy right issues, we are not
able to publish this dataset at this time. However, we have included a demo dataset from the famous movilens
data. This program has various methods to create group data. This program uses demo dataset as movilens. It
groups together a set of users who saw same set of movies. This decision is done on various criteria. 
'''

from django.utils.encoding import smart_str
import json
import re
import unicodedata
import numpy as np
import gc
import time
import sys
import os
from scipy.stats.stats import pearsonr
from copy import deepcopy
import linecache

dest_folder2 = 'movlens'
toolbar_width = 0

def progressbar(set='start'):
    
    if(set == 'start'):
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    elif(set == 'continue'):
        time.sleep(0.1) # do real work here
        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()
    else:
        sys.stdout.write("\n")

'''calculate the pearson correlation for group members and 
returns the groups based on the desired threshold of correlation'''
def calculateCorrelation(grp_data,orig_usr_id,grp_sz,dat_typ):
    
    if(dat_typ == 'movi'):
        data_loc = dest_folder2+'/mov_n_ratings.txt'
        writ_fp = open(dest_folder2+'/pcc_movlens_grpsz'+str(grp_sz)+'.csv','w')

    dat_file = open(data_loc,'r')
    user_id1 = set()
    itm_id1 = set()
    for line in dat_file:
        line = line.strip()
        line = line.split('::')
        usr = int(line[0])
        user_id1.update([usr])
        itm = int(line[1])
        itm_id1.update([itm])
    dat_file.seek(0)
    # create the user id matrix
    usr_itm_mat = np.zeros([max(user_id1)+1,max(itm_id1)+1])
    for line in dat_file:
        line = line.strip()
        line = line.split('::')
        usr = int(line[0])
        itm = int(line[1])
        if(len(line) > 2):
            rating = int(line[2])
        else: 
            rating = 1
        usr_itm_mat[usr,itm] = rating
    print 'finding pearson corr for %d groups'%(len(grp_data))
    # finding pearson corr for every group
    for cnt,grps in enumerate(grp_data):
        # iterate through every usr in group and find PRR
        tot_pcc = []
        t_grp = deepcopy(grps)
        for s_indx, usr_itm1 in enumerate(grps):
            s_usr = usr_itm1[0]
#             s_usr = orig_usr_id[s_usr]
            for usr_itm2 in t_grp:
                # skip if user is same as the source user
                t_usr = usr_itm2[0]
                if (s_usr == t_usr):
                    continue
                else:
#                     t_usr = orig_usr_id[t_usr]
                    # get the itm rating/baking history of users
                    s_usr_hist = usr_itm_mat[int(s_usr)]
                    t_usr_hist = usr_itm_mat[int(t_usr)]
                    #calculate the pearson corr between users
                    tot_pcc.append(pearsonr(s_usr_hist,t_usr_hist)[0])
            # remove the user who is finished comparing
            del t_grp[0]
        avg_pcc = np.round(np.average(tot_pcc),2)
        writ_fp.write(str(cnt)+','+str(avg_pcc)+'\n')
        writ_fp.flush()
        print 'groups %d done'%(cnt)


# method to filter groups based on group size and common items liked 
def filterKckGroups(grp_sz=1,itm_sz=1,print_stat='no',data_typ='movi'):           
    
    
    if (data_typ == 'movi'):
        mov_lens = '/movlens_grps_szComplete.json'
        orig_usrid_f = open(dest_folder2+'/orig_usrid_grpsz'+str(grp_sz)+'itmsz_'+str(itm_sz)+'.csv','w')
        orig_itmid_f = open(dest_folder2+'/orig_itmid_grpsz'+str(grp_sz)+'itmsz_'+str(itm_sz)+'.csv','w')
        fp = open(dest_folder2+'/movilensGroup_sz'+str(grp_sz)+'_itm'+str(itm_sz)+'.csv','w')
        with open(dest_folder2+mov_lens,'r') as fopen:
            data = json.load(fopen)
        print "creating movie lens group dataset"
    
    itm_sizes = []
    usrs_per_grp = []
    grp_ids = dict()
    ''' These dictionaries have the ORIGINAL id of the user and item as VALUE
    and the KEYS contain the fake ids with corresponds to ids for a specific filtered group'''
    usr_ids = dict()
    itm_ids = dict()
    orig_usr_ids = dict()
    orig_itm_ids = dict()
    grps_n_bking = []
    
    # calculate avg items per group
    print 'program will be analyzing %d groups...starting'%(len(data))
    time.sleep(7)
    count = 0
    for grps in data:
        count += 1
        print 'done with %d groups'%(count)
        # the keys are actually in list, but json stores them as strings. So we have to convert list to string
        pr_grps = re.sub("'",'',grps)
        pr_grps = grps.strip('(|)')
        pr_grps  = unicodedata.normalize('NFKD', pr_grps).encode('ascii','ignore')
        no_of_usr = pr_grps.split(',')
        no_of_usr = [re.sub("'",'',i).strip() for i in no_of_usr]
        if('' in no_of_usr):
            no_of_usr.remove('')
        items = data[grps][0]
        if(len(no_of_usr) >= grp_sz and len(items) >= itm_sz):
            # append empty list to create the data for groups
            grps_n_bking.append([])
            # id entry for the groups
            if (grps not in grp_ids):
                grp_ids[grps] = len(grp_ids)
            # write group to file
            fp.write(str(grp_ids[grps])+','+'\t')
                     
            # id entry for users
            for i in no_of_usr:
                if (i not in usr_ids):
                    orig_usrid_f.write(str(len(usr_ids))+','+str(i)+'\n')
                    orig_usrid_f.flush()
                    usr_ids[i] = len(usr_ids) 
#                             orig_usr_ids[str(len(usr_ids))] = i
                       
            # create ids for items
            for i in items:
                ''' for some reason the json file has project ids stored in 
                unicode format so remove those unicode using django smartstring'''
                i = smart_str(i)
                if (i not in itm_ids):
                    orig_itmid_f.write(str(len(itm_ids))+','+str(i)+'\n')
                    itm_ids[i] =  len(itm_ids)
#                             orig_itm_ids[str(len(itm_ids))] = i
                          
                      
            # make the list of groups and its backing
            indx_entry = len(grps_n_bking)-1
            for i in no_of_usr:
                i = usr_ids[i]
                for j in items:
                    j = smart_str(j)
                    j = itm_ids[j]
                    fp.write(str(i)+','+str(j)+'\t')
                    bkr_itm = (i,j)
                    grps_n_bking[indx_entry].append(bkr_itm)
            fp.write('\n')
            usrs_per_grp.append(len(no_of_usr))
            itm_sizes.append(int(data[grps][1]))   
    
    #Recreate backer and project pair 
    map_ids(usr_ids,itm_ids,dat_type=data_typ)
    
    def print_stats():        
        print "******following are the statistics for a group size of atleast %d*******"%(grp_sz)
        print 'total groups %d'%(len(grp_ids))
        print 'total unique users %d'%(len(usr_ids))
        print 'average users per group %d'%(np.average(usrs_per_grp))
        print 'median users per group %d'%(np.median(usrs_per_grp))
        print 'total unique items %d'%(len(itm_ids))
        print 'median number of project that are commonly backed by a group %d'%(np.median(itm_sizes))
        print 'average number of projects that are commonly backed by a group %d'%(np.average(itm_sizes))
        print 'maximum number of projects that are commonly backed by a group %d'%(np.max(itm_sizes))
    if(print_stat != 'no'):
        print_stats()
#         calculateCorrelation(grp_data=grps_n_bking,orig_usr_id=orig_usr_ids,orig_itm_id=orig_itm_ids)
#         return(grps_n_bking,grp_ids,usr_ids,itm_ids)     


#This function maps new user ids and item ids to the old ones
def map_ids(usr_ids,itm_ids,dat_type):

    dat_file2 = open(dest_folder2+'/mov_n_ratings.txt','w')
    dat_file = open(dest_folder2+'/movielens_ratings.dat','r')
        
    for line in dat_file:
        line = line.strip()
        line = line.split('::')
        usr = line[0]
        itm=line[1]
        if usr in usr_ids:
            newid = usr_ids[usr]
            if itm in itm_ids:
                newid2 = itm_ids[itm]
                dat_file2.write(str(newid)+'::'+str(newid2)+'::1\n')
    dat_file2.close()
    dat_file.close()


# method to fetch the pearson correlation co-eff
def fetch_prr(grp_sz=2,data_typ='movi',fetch='force'):
        
        
        if(data_typ == 'movi'):
            if(fetch != 'force'):
                if (os.path.isfile(dest_folder2+'/pcc_movlens_grpsz'+str(grp_sz)+'.csv')):
                    print 'file alread exist no need to calculate'
                    sys.exit(1)
            # if you need movie lens data
            read_file0 = open(dest_folder2+'/orig_usrid_grpsz'+str(grp_sz)+'itmsz_'+str(grp_sz)+'.csv','r')
            read_file1 = open(dest_folder2+'/movilensGroup_sz'+str(grp_sz)+'_itm'+str(grp_sz)+'.csv','r')

        grp_data = []
        usr_ids = dict()
        for line in read_file0:
            line = line.strip()
            line = line.split(',')
            usr_ids[line[0]] = line[1]
        for line in read_file1:
            line = line.strip()
            rd_line = line.split('\t')
            grp_data.append([])
            index = len(grp_data)-1
            for val in rd_line[1:]:
                val = val.split(',')
                usr_itm = (val[0],val[1])
                grp_data[index].append(usr_itm)

        calculateCorrelation(grp_data=grp_data,orig_usr_id=usr_ids,grp_sz=grp_sz,dat_typ=data_typ)

def fetch_grp_data(grp_sz=5,itm_sz=5,pcc_cutoff=0.45,data_nam='movi'):
    
    list_grpids=[]
    list_pcc=[]
    grp_data = []
    new_grp = set()
    new_usrid = dict()
    new_itmid = dict()
    orig_usrid = dict()
    orig_itmid = dict()
    if (data_nam=='movi'):
        file1=open(dest_folder2+'/pcc_movlens_grpsz'+str(grp_sz)+'.csv',"r")

    for line in file1:
        line = line.strip()
        line=line.split(",")
        if float(line[1])>pcc_cutoff:
            list_grpids.append(line[0])
            list_pcc.append(line[1])
    file1.close()
    
    if (data_nam=='movi'):
        file2=dest_folder2+'/movilensGroup_sz'+str(grp_sz)+'_itm'+str(itm_sz)+'.csv'
        file3 = open(dest_folder2+'/pcc_filteredgroup_sz'+str(grp_sz)+'itm'+str(itm_sz)+'.csv','w')

    for grpid in list_grpids:
        # create new group id for these filtered list
        if(grpid not in new_grp):
            new_gid = len(new_grp)
            new_grp.add(str(new_gid))
            file3.write(str(new_gid)+'\t')
        # go the respective filtered group's file location
        line = linecache.getline(file2,int(grpid)+1)
        line=line.strip()
        rd_line=line.split("\t")
        grp_data.append([])
        index = len(grp_data)-1
        for val in rd_line[1:]:
            val = val.split(',')
            usr=int(val[0])
            itm=int(val[1])
            if (usr not in new_usrid):
                nusr_id = len(new_usrid)
                new_usrid[usr] = nusr_id
#                 orig_usrid[nusr_id] = usr
            file3.write(str(new_usrid[usr])+',')
            if (itm not in new_itmid):
                nitm_id = len(new_itmid)
                new_itmid[itm] = nitm_id
#                 orig_itmid[nitm_id] = itm
            file3.write(str(new_itmid[itm])+'\t')
            grp_data[index].append((new_usrid[usr],new_itmid[itm]))
        file3.write('\n')
    print 'totgrps:%d, totusrs:%d, totitms:%d'%(len(grp_data),len(new_usrid),len(new_itmid))
    return(grp_data,new_usrid,new_itmid)
    
if __name__ == '__main__':
    grp_sz =5
    itm_sz = 5
    dat_typ='movi'
    filterKckGroups(grp_sz=grp_sz,itm_sz=itm_sz, print_stat='yes',data_typ=dat_typ)
    fetch_prr(grp_sz=grp_sz,data_typ=dat_typ,fetch='force')
#     fetch_grp_data(grp_sz=grp_sz,itm_sz=itm_sz,pcc_cutoff=0.35,data_nam=dat_typ)
