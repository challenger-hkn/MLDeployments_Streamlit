import streamlit as st
import pickle
import pandas as pd
import base64
from IPython.core.display import HTML
from PIL import Image

st.set_page_config(
    page_title='Employee Decision Predictor',
    page_icon='icon1.png'
	)

st.markdown(
			"<h2 style='font-size:230%;\
						margin-bottom: -25px;\
						font-family:tahoma;\
						border-radius: 10px;\
						text-align:center;\
						background-color:lavender;\
						color:indigo;'>EMPLOYEE LEAVE PREDICTION \
						\
			</h2>", unsafe_allow_html=True)

# st.write('\n')

st.markdown("""
			<style>
    		[data-baseweb="select"] {margin-top: -40px;}
    		</style>
    		""", unsafe_allow_html=True,)

st.sidebar.markdown(
			"<h1 style='text-align:center;\
						color: red;font-family:tahoma;font-size:115%;'> Human is the most valuable asset.\
						Retain your employees.\
			</h1>", unsafe_allow_html=True)

# st.sidebar.image('churn.png', width=300)
st.sidebar.markdown(
    		f"""
    		<div>
        	<img class="churn1.png" 
				src="https://www.reminetwork.com/wp-content/uploads/employeechurn.jpg" 
				width="300">
			</img>
    		</div>
    		""", unsafe_allow_html=True)

st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:lightblue;'>Please Enter Employee Information\
			</p>
			""", unsafe_allow_html=True)

Satisfaction_Level = st.sidebar.slider(label = "Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01, )
Last_Evaluation = st.sidebar.slider(label="Last Evaluation", min_value=0.0, max_value=1.0, step=0.01)
Time_Spend_Company = st.sidebar.slider("Time Spend in the Company", min_value=0, max_value=15, step=1)
Average_Monthly_Hours = st.sidebar.slider("Average Monthly Hours", min_value=0, max_value=350, step=1)

st.sidebar.write('\n')

Number_Project = st.sidebar.number_input(label="Number of Projects", min_value=1, max_value=10)

st.sidebar.write('\n')

st.sidebar.markdown("""<p style='text-align:left;color:black;font-size:90%;'>Departments</p>""", unsafe_allow_html=True)
# st.sidebar.write('\n')
Departments = st.sidebar.selectbox("", ['IT','RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng',  'sales', 'support', 'technical'])

st.sidebar.markdown("""<p style='text-align:left;color:black;font-size:90%;'>Salary</p>""", unsafe_allow_html=True)
# st.sidebar.write('\n')
Salary = st.sidebar.selectbox("", ['low', 'medium', 'high'])

Work_Accident = st.sidebar.radio("Work Accident", ("Yes", "No"))
if Work_Accident=="Yes":
    Work_Accident=1
else:
    Work_Accident=0	
	
Promotion_Last_5Years = st.sidebar.radio("Promotion in Last 5 Years", ("Yes", "No"))
if Promotion_Last_5Years=="Yes":
    Promotion_Last_5Years=1
else:
    Promotion_Last_5Years=0	


coll_dict = {'Satisfaction_Level':Satisfaction_Level, 'Last_Evaluation':Last_Evaluation, \
			'Number_Project':Number_Project, "Average_Montly_Hours":Average_Monthly_Hours,\
			'Time_Spend_Company':Time_Spend_Company, 'Work_Accident':Work_Accident, \
			'Promotion_Last_5Years':Promotion_Last_5Years,\
			'Salary':Salary, 'Departments': Departments}

columns = ['Satisfaction_Level', 'Last_Evaluation', 'Number_Project',
       'Average_Montly_Hours', 'Time_Spend_Company', 'Work_Accident',
       'Promotion_Last_5Years', "Salary", "Departments_IT",'Departments_RandD',
       'Departments_accounting', 'Departments_hr', 'Departments_management',
       'Departments_marketing', 'Departments_product_mng',
       'Departments_sales', 'Departments_support', 'Departments_technical']

st.write('\n')
st.markdown("""
			<center>
			<p style='font-size:140%;\
						background-color:lavender;\
						border-radius:10px;\
						color:indigo;'>Employee Information\
			</p>
			</center>
			""", unsafe_allow_html=True)

df_show = pd.DataFrame.from_dict([coll_dict])
df_show = pd.DataFrame(df_show, index=[0])
df_show = df_show.rename(columns={"Satisfaction_Level":"Satisfaction Level",
              "Last_Evaluation":"Last Evaluation",
              "Number_Project":"Number Projects",
              "Average_Montly_Hours":"Monthly Hours",
              "Time_Spend_Company":"Years in Company",
              "Work_Accident":"Work Accident",
              "Promotion_Last_5Years":"Received Promotion",
              "Departments":"Departments",
              "Salary":"Salary"},)
df_show['Monthly Hours'] = df_show['Monthly Hours'].astype('int')
df_show['Work Accident'] = df_show['Work Accident'].map({0:'No', 1:'Yes'})
df_show['Received Promotion'] = df_show['Received Promotion'].map({0:'No', 1:'Yes'})
df_show['Departments'] = df_show['Departments'].map({'IT':'Information Technology',
                        'RandD': 'R & D',
                        'accounting':'Accounting',
                        'hr':'Human Resources',
                        'management':'Management',
                        'marketing':'Marketing',
                        'product_mng':'Product Management',
                        'sales':'Sales',
                        'support':'Support',
                        'technical':'Technical'})
df_show['Salary'] = df_show['Salary'].map({'low':'Low', 'medium':'Medium', 'high':'High'})

st.write('\n')

st.write((HTML(df_show.to_html(index=False, justify='left'))))

# dumy model

df_input = pd.DataFrame.from_dict([coll_dict])
df_input.Salary = df_input.Salary.map({"low":1, "medium" : 2, "high" : 3})
scaler= pickle.load(open("scaler_knn.pkl", 'rb'))
df_input_dumy = pd.get_dummies(df_input).reindex(columns=columns, fill_value=0)
st.write(df_input)
st.write(df_input_dumy)
df_input_scaled = scaler.transform(df_input_dumy)
st.write(df_input_scaled)
# encoder
loaded_enc = pickle.load(open("encoder.pkl", 'rb'))
new_df = pd.DataFrame(df_input, index=[0])

cat = new_df.select_dtypes("object").columns
new_df[cat] = loaded_enc.transform(new_df[cat])


st.sidebar.markdown("""
			<p style='text-align:center;\
						color: white; background-color:lightblue;'>Please complete the  employee information\
			</p>
			""", unsafe_allow_html=True)
st.write('\n')

st.markdown("""
    		<h3 style='font-size:130%;\
						font-family:thoma;\
						background-color:;\
						text-align: left\
						border-radius: px;\
						color:darkblue;'>Select Your Model\
			</h3>
			""", unsafe_allow_html=True)

st.markdown("""
			<p style='font-size:90%;\
			font-family:tahoma;\
			border-radius: 2px;\
			color:white;'> Select one model and get your prediction\
			</p>
			""", unsafe_allow_html=True)

selection = st.selectbox("",
			["Gradient Boosting", "Random Forest", "KNN"])

if selection =="Gradient Boosting":
	model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
	prediction= model.predict(new_df)

elif selection =="Random Forest":
	model = pickle.load(open('rf_grid_model.pkl', 'rb'))
	prediction = model.predict(new_df)
elif selection =="KNN":
	model = pickle.load(open('knn_final.pkl', 'rb'))
	prediction = model.predict(df_input_scaled)

# st.write('\n')

col1, col2, = st.columns([1, 1.5])
if col2.button("PREDICT"):
	if prediction[0]==0:
		st.success(f'The employee will **stay** with the team :)')
		st.markdown(f"""
    		<div>
        	<img class="group3.png" 
				 src="data:image/png;base64,{base64.b64encode(open("to be continued1.png", "rb").read()).decode()}" 
				 width="704">
			</img>
    		</div>
    		""", unsafe_allow_html=True)		
	elif prediction[0]==1:
		# st.warning(prediction[0])
		st.warning(f'The employee will **leave** with the team :(')

		st.markdown(f"""
    		<div>
        	<img class="group3.png" 
				 src="data:image/png;base64,{base64.b64encode(open("left.png", "rb").read()).decode()}" 
				 width="704">
			</img>
    		</div>
    		""", unsafe_allow_html=True)

# To hide Streamlit style
hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

# to add a background image
@st.cache(allow_output_mutation=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;}
        </style>
    """ % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg('background.png')
