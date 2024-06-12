from transformers import pipeline
import pandas as pd

# Load the text correction model
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
print('Model downloaded!!!')

# Load the comments from a CSV file
vietnam_comment = pd.read_csv('database/vietnam_comment.csv')
print('Dataframe inputed!!!')

# Ensure all comments are strings
texts = vietnam_comment['comment'].astype(str).tolist()

print('Texts converted!!!')

# Correct the text comments
print('Predicting!!!')
predictions = corrector(texts, max_length=512)

# Add the corrected comments to the DataFrame
vietnam_comment['edited_comment'] = [prediction['generated_text'] for prediction in predictions]

# Save the updated DataFrame to a new CSV file
vietnam_comment.to_csv('database/updated_vietnam_comment.csv', encoding='utf-8-sig',index=False)
print('DONE!!')