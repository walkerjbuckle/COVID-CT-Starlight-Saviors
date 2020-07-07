#!/usr/bin/env python
# coding: utf-8

# In[15]:




# In[4]:


get_ipython().system("python main_moco.py  -a densenet169   --data ../../Images-processed --lr 0.015   --batch-size "
                     "32   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed 1 --world-size 1 --rank 0 "
                     " --epochs 5")
# --resume save_model_dense/checkpoint_covid.pth.tar
#--data ../../Images-processed

# In[ ]:




