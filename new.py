from taipy import Gui
import matplotlib.pyplot as plt
from running import node

title="Epidemiological Simulator Using SEIQVR Model"
title2="Select Parameters For Each Region:"
sl1="Gamma Mortality1: "
sl2="Gamma Mortality2: "
sl3="Gamma Immunized: "
int_exp = 2
qr=50
hospital_cap=728
gamma_mor1=3
gamma_mor2=4.3
gamma_imm=83
data=""
def button_pressed(state):
  data = node.df.columns.difference(['sus'])
  print(data)
  '''plt.figure(figsize=(8, 8))
# Plot each column against the index
  for col in cols:
      plt.plot(node.df.index, node.df[col])

# Add legend and show plot
  plt.legend(cols)
  plt.show()'''

page="""
<|text-center|
<|{title}|>
>

<|text-left|
<|{title2}|>
>

Initial Exp: <|{int_exp}|input|>
Daily Quarentine Rate: <|{qr}|input|>
Hospital Capacity: <|{hospital_cap}|input|>

<|{sl1}|>

<|{gamma_mor1}|slider|min=0|max=100|>

<|{sl2}|>

<|{gamma_mor2}|slider|min=0|max=100|>

<|{sl3}|>

<|{gamma_imm}|slider|min=0|max=100|>

<|Start Simulation|button|on_action=button_pressed|>

<|{node.df}|chart|mode=lines|x=index|y[1]=con|y[2]=dea|y[3]=exp|y[4]=inf|y[5]=imm|y[6]=iso|y[7]=qua|y[8]=vac|line[1]=cyan|color[2]=grey|color[3]=red|color[4]=purple|color[5]=brown|color[6]=pink|color[7]=yellow|color[8]=blue|>


"""


if __name__=="__main__":
    app = Gui(page)
    app.run(use_reloader=True)