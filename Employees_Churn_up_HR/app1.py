from tkinter.messagebox import YES
import streamlit as st
import pickle
import pandas as pd
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier

#st.title('Employee Leave Prediction')
st.markdown(
			"<h1 style='font-size:400%;\
						font-family:times;\
						text-align:center;\
						background-color:;\
						color:blue;'>EMPLOYEE LEAVE PREDICTION\
						GROUP 3\
			</h1>", unsafe_allow_html=True
			)
st.markdown("""
			<div style="background-color:orange;\
						border-radius: 10px;\
						padding:15px">
			<h2 style="color:white;\
					   text-align:center;\
					   font-family:cursive;">Streamlit Churn Prediction ML App\
			</h2>
			</div>
			""", unsafe_allow_html=True
			)
st.write('\n')

st.markdown(
    		f"""
    		<div>
        	<img class="group3.png" 
				 src="data:image/png;base64,{base64.b64encode(open("group3.png", "rb").read()).decode()}" 
				 width="704">
			</img>
    		</div>
    		""", unsafe_allow_html=True
			)
st.write('\n')
st.markdown("""
			<center>
			<p style='font-size:200%;\
						font-family:cursive;\
						background-color:orange;\
						border-radius: 10px;\
						color:white;'>Select Your ML Model\
			</p>
			</center>
			""", unsafe_allow_html=True
			)

st.markdown("""
			<style>
    		[data-baseweb="select"] {
        							margin-top: -50px;
    								}
    		</style>
    		""", unsafe_allow_html=True,
			)

# selection = st.selectbox("",["Gradient Boost", "Random Forest", "KNN"])

# if selection =="Gradient Boost":
# 	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
# 				You selected \
# 				<span style='color:tomato;font-weight:bold'>\
# 				'Gradient Boost'\
# 				</span> model!\
# 				</p>", unsafe_allow_html=True
# 				)
# 	grad_model = pickle.load(open('grad_model.pkl', 'rb'))

# elif selection =="Random Forest":
# 	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
# 				You selected \
# 				<span style='color:tomato;font-weight:bold'>\
# 				'Random Forest'\
# 				</span> model!\
# 				</p>", unsafe_allow_html=True
# 				)
# 	RF_model = pickle.load(open('random_forest_model.pkl', 'rb'))

# elif selection =="KNN":
# 	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
# 				You selected \
# 				<span style='color:tomato;font-weight:bold'>\
# 				'KNN'\
# 				</span> model!\
# 				</p>", unsafe_allow_html=True
# 				)
# 	KNN_model = pickle.load(open('knn_final.pkl', 'rb'))
st.sidebar.markdown(
			"<h1 style='text-align:center;\
						color: tomato;font-family:cursive;font-size:115%;'>Will Your Employee Run Away\
			</h1>", unsafe_allow_html=True
			)

st.sidebar.markdown(
    		f"""
    		<div>
        	<img class="emir.png" 
				 src="data:image/png;base64,{base64.b64encode(open("churn.png", "rb").read()).decode()}" 
				 width="300">
			</img>
    		</div>
    		""", unsafe_allow_html=True
			)

st.sidebar.write('\n')
st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:orange;'>Please Slide\
			</p>
			""", unsafe_allow_html=True
			)
satisfaction_level = st.sidebar.slider(label = "Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01, )
last_evaluation = st.sidebar.slider(label="Last Evaluation", min_value=0.0, max_value=1.0, step=0.01)
time_spend_company = st.sidebar.slider("Time Spend in the Company", min_value=0, max_value=30, step=1)
average_monthly_hours = st.sidebar.slider("Average Monthly Hours", min_value=0, max_value=350, step=1)

st.sidebar.write('\n')
st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:orange;'>Please Choose\
			</p>
			""", unsafe_allow_html=True
			)
Work_accident = st.sidebar.radio("Work Accident", ("Yes", "No"))
if Work_accident=="Yes":
    Work_accident=1
else:
    Work_accident=0	
promotion_last_5years = st.sidebar.radio("Promotion in Last 5 Years", ("Yes", "No"))
if promotion_last_5years=="Yes":
    promotion_last_5years=1
else:
    promotion_last_5years=0	

st.sidebar.write('\n')
st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:orange;'>Please Use "+" and "-" Buttons\
			</p>
			""", unsafe_allow_html=True
			)
number_project = st.sidebar.number_input(label="Number of Projects", min_value=1, max_value=200)

st.sidebar.write('\n')
st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:orange;'>Please Select From List\
			</p>
			""", unsafe_allow_html=True
			)
st.sidebar.markdown("""<p style='text-align:left;color:black;font-size:90%;'>Departments</p>""", unsafe_allow_html=True)
st.sidebar.write('\n')
Departments = st.sidebar.selectbox("Departments", ['RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng',  'sales', 'support', 'technical', 'IT'])

st.sidebar.markdown("""<p style='text-align:left;color:black;font-size:90%;'>Salary</p>""", unsafe_allow_html=True)
st.sidebar.write('\n')
salary = st.sidebar.selectbox("Salary", ['low', 'medium', 'high'])


coll_dict = {'satisfaction_level':satisfaction_level, 'last_evaluation':last_evaluation, 'number_project':number_project, 'average_montly_hours':average_monthly_hours,\
			'time_spend_company':time_spend_company, 'Work_accident':Work_accident, 'promotion_last_5years':promotion_last_5years,\
			'Departments': Departments, 'salary':salary}
columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',\
            'Work_accident', 'promotion_last_5years', 'Departments_RandD', 'Departments_accounting', 'Departments_hr',\
            'Departments_management', 'Departments_marketing', 'Departments_product_mng', 'Departments_sales',\
            'Departments_support', 'Departments_technical', 'salary_low', 'salary_medium']

# df = pd.read_csv("HR_Dataset.csv")
# df.rename(columns={"Departments_": "Departments"}, inplace=True)
# df1=df.drop(columns='left')

# dumy model
df_input = pd.DataFrame.from_dict([coll_dict])
scaler= pickle.load(open("scaler.pkl", 'rb'))
# scaler.clip = False
user_inputs_dumy = pd.get_dummies(df_input).reindex(columns=columns, fill_value=0)
user_inputs_transformed = scaler.transform(user_inputs_dumy)

# encoder
loaded_enc = pickle.load(open("encoder_son.pkl", 'rb'))
new_df = pd.DataFrame(df_input, index=[0])
new_df.salary = new_df.salary.map({"low":1, "medium" : 2, "high" : 3})

cat = new_df.select_dtypes("object").columns
new_df[cat] = loaded_enc.transform(new_df[cat])
st.write(new_df)



selection = st.selectbox("",["Gradient Boost", "Random Forest", "KNN"])
if selection =="Gradient Boost":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:tomato;font-weight:bold'>\
				'Gradient Boost'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
	prediction= model.predict(new_df)

elif selection =="Random Forest":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:tomato;font-weight:bold'>\
				'Random Forest'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('rf_grid_model.pkl', 'rb'))
	prediction = model.predict(new_df)
elif selection =="KNN":
	st.markdown("<p style='text-align:center; color:black; font-size:110%; background-color:#F2F3F4 ;'>\
				You selected \
				<span style='color:tomato;font-weight:bold'>\
				'KNN'\
				</span> model!\
				</p>", unsafe_allow_html=True
				)
	model = pickle.load(open('knn_final_pickle.pkl', 'rb'))
	prediction = model.predict(user_inputs_transformed)

st.sidebar.markdown(
			"<h1 style='text-align:center;\
						color: tomato;font-family:cursive;font-size:115%;'>Will Your Employee Run Away\
			</h1>", unsafe_allow_html=True
			)



# if selection =="KNN":
# 	user_inputs = pd.get_dummies(df_coll).reindex(columns=columns, fill_value=0)
# 	scaler= pickle.load(open("scaler_knn.pkl", 'rb'))
# 	scaler.clip = False
# 	user_inputs_transformed = scaler.transform(user_inputs)
# 	prediction = KNN_model.predict(user_inputs_transformed)



st.markdown("""
			<center>
			<h1 style='font-size:200%;\
						background-color:orange;\
						border-radius: 10px;\
						color:white;'>Employee Information\
			</h1>
			</center>
			""", unsafe_allow_html=True
			)

st.write('\n')

st.dataframe(new_df)

st.markdown("""
			<center>
			<p style='font-size:200%;\
						font-family:cursive;\
						background-color:orange;\
						border-radius: 10px;\
						color:white;'>Click PREDICT if configuration is OK\
			</p>
			</center>
			""", unsafe_allow_html=True
			)


col1, col2, = st.columns([1, 1.5])
if col2.button("PREDICT"):
	if prediction[0]==0:
		st.success(prediction[0])
		st.success(f'Employee will STAY :)')
	elif prediction[0]==1:
		st.warning(prediction[0])
		st.warning(f'Employee will LEAVE :(')


# enc = OrdinalEncoder()
# cat = use_df.select_dtypes("object").columns
# use_df[cat]= enc.fit_transform(use_df[cat])
# new_df = use_df[:1]

# loaded_model= pickle.load(open("xgb_final.pkl", 'rb'))
# prediction= loaded_model.predict(new_df)
# prediction_price=int(prediction)
# new_df[new_df.select_dtypes('object').columns] = loaded_enc.transform(new_df[new_df.select_dtypes('object').columns])
# st.write(new_df)

# use_df = pd.concat([df_input, df1], axis=0)
# st.write(use_df)
# st.write(df_input)

# enc = OrdinalEncoder()
# cat = use_df.select_dtypes("object").columns
# use_df[cat]= enc.fit_transform(use_df[cat])
# # st.write(use_df)
# new_df = use_df[:1]
# st.write(df_input)