from sqlalchemy import create_engine
import pandas as pd


df = pd.read_csv('./penguins_lter.csv')
engine = create_engine()