# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:18:37 2024

@author: nehak
"""

# ======================================================================================================================
# Set batch size and file names in which pre-processed data will be saved
# ======================================================================================================================


file = 'data\\kp20k_validation.json' 



# VALIDATION data path
x_filename = 'data\\preprocessed_data\\x_VALIDATION_data_preprocessed'
y_filename = 'data\\preprocessed_data\\y_VALIDATION_data_preprocessed'



#  TEST data - use for EVALUATION (exact/partial matching)
x_text_filename = 'data\\preprocessed_data\\x_TEST_preprocessed_TEXT'  
y_text_filename = 'data\\preprocessed_data\\y_TEST_preprocessed_TEXT'  


batch_size = 32  