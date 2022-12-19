# imports
import pickle

import pandas            as pd
import numpy             as np
import xgboost           as xgb

from category_encoders       import TargetEncoder
from datetime                import datetime
from dateutil                import relativedelta
from sklearn.preprocessing   import RobustScaler, MinMaxScaler
from skopt                   import gp_minimize
from pandas_profiling        import ProfileReport

from sklearn                 import model_selection   as ms

# class
class Datarisk_predictor( object ):
    def diff_month(self, d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def diff_days(self, d1, d2):
        return (d1.year - d2.year) * 365 + d1.month - d2.month



    def data_cleaning( self, df ):
        df['DATA_EMISSAO_DOCUMENTO'] = pd.to_datetime( df['DATA_EMISSAO_DOCUMENTO'] )
        df['DATA_VENCIMENTO'] = pd.to_datetime( df['DATA_VENCIMENTO'] )
        df['DATA_CADASTRO'] = pd.to_datetime( df['DATA_CADASTRO'], errors='coerce' )

        df['DDD'] = df['DDD'].apply(lambda x: x.replace("(", "") if type(x) is not float else x)
        df['DDD'] = df['DDD'].astype(float)

        df['CEP_2_DIG'] = df['CEP_2_DIG'].apply( lambda x: np.nan if x=="na" else x)
        df['CEP_2_DIG'] = df['CEP_2_DIG'].astype(float)

        df['FLAG_PF'] = df['FLAG_PF'].apply( lambda x: 1 if x=='X' else 0)

        df['RENDA_MES_ANTERIOR'] = df.groupby(['ID_CLIENTE']).ffill()['RENDA_MES_ANTERIOR']
        df['RENDA_MES_ANTERIOR'] = df['RENDA_MES_ANTERIOR'].fillna(0)

        df['NO_FUNCIONARIOS'] = df.groupby(['ID_CLIENTE']).ffill()['NO_FUNCIONARIOS']
        df['NO_FUNCIONARIOS'] = df['NO_FUNCIONARIOS'].fillna(0)

        df = df.fillna(0)

        return df



    def feature_engineering( self, df ):
        df['prazo_em_dias'] = df['DATA_VENCIMENTO'] - df['DATA_EMISSAO_DOCUMENTO']
        df['prazo_em_dias'] = df['prazo_em_dias'].dt.days

        df['meses_desde_cadastro'] = df.apply( lambda x: self.diff_month(x.DATA_EMISSAO_DOCUMENTO, x.DATA_CADASTRO) if type(x.DATA_CADASTRO) is not int else 0, axis=1)
        df['dias_desde_cadastro'] = df.apply( lambda x: self.diff_days(x.DATA_EMISSAO_DOCUMENTO, x.DATA_CADASTRO) if type(x.DATA_CADASTRO) is not int else 0, axis=1)

        df['valor_emprestimo'] = df.apply( lambda x: x['VALOR_A_PAGAR'] / (1 + x['TAXA'] / 100), axis=1 )

        df['diff_renda'] = df['RENDA_MES_ANTERIOR'] - df['VALOR_A_PAGAR']

        df['len_credit_history'] = np.nan
        for id in df['ID_CLIENTE'].unique():
            len_credit_history = len(df.loc[df['ID_CLIENTE']==id])
            df.loc[df['ID_CLIENTE']==id, 'len_credit_history'] = len_credit_history

        df = df.assign(ultima_data_emprestimo=df.groupby('ID_CLIENTE').DATA_EMISSAO_DOCUMENTO.apply(lambda x: x.diff().dt.days))
        df['ultima_data_emprestimo'] = df['ultima_data_emprestimo'].fillna(0)

        return df



    def data_preparation( self, df ):
        
        # encoding
        df['SAFRA_REF'] = df['SAFRA_REF'].astype(str)
        df['DATA_EMISSAO_DOCUMENTO'] = df['DATA_EMISSAO_DOCUMENTO'].astype(str)
        df['DATA_VENCIMENTO'] = df['DATA_VENCIMENTO'].astype(str)
        df['DATA_CADASTRO'] = df['DATA_CADASTRO'].astype(str)

        data_cadastro_encoding = pickle.load( open( 'encoders/data_cadastro_encoding', 'rb' ) )
        df['DATA_CADASTRO'] = data_cadastro_encoding.transform(df['DATA_CADASTRO'])

        data_emissao_documento_encoding = pickle.load( open( 'encoders/data_emissao_documento_encoding', 'rb' ) )
        df['DATA_EMISSAO_DOCUMENTO'] = data_emissao_documento_encoding.transform(df['DATA_EMISSAO_DOCUMENTO'])

        data_vencimento_encoding = pickle.load( open( 'encoders/data_vencimento_encoding', 'rb' ) )
        df['DATA_VENCIMENTO'] = data_vencimento_encoding.transform(df['DATA_VENCIMENTO'])

        dominio_email_encoding = pickle.load( open( 'encoders/dominio_email_encoding', 'rb' ) )
        df['DOMINIO_EMAIL'] = dominio_email_encoding.transform(df['DOMINIO_EMAIL'])

        porte_encoding = pickle.load( open( 'encoders/porte_encoding', 'rb' ) )
        df['PORTE'] = porte_encoding.transform(df['PORTE'])

        safra_ref_encoding = pickle.load( open( 'encoders/safra_ref_encoding', 'rb' ) )
        df['SAFRA_REF'] = safra_ref_encoding.transform(df['SAFRA_REF'])

        segmento_industrial_encoding = pickle.load( open( 'encoders/segmento_industrial_encoding', 'rb' ) )
        df['SEGMENTO_INDUSTRIAL'] = segmento_industrial_encoding.transform(df['SEGMENTO_INDUSTRIAL'])

        # rescaling
        TAXA_scaler = pickle.load( open( 'scalers/TAXA_scaler', 'rb' ) )
        df['TAXA'] = TAXA_scaler.transform(df[['TAXA']].values)

        NO_FUNCIONARIOS_scaler = pickle.load( open( 'scalers/NO_FUNCIONARIOS_scaler', 'rb' ) )
        df['NO_FUNCIONARIOS'] = NO_FUNCIONARIOS_scaler.transform(df[['NO_FUNCIONARIOS']].values)

        prazo_em_dias_scaler = pickle.load( open( 'scalers/prazo_em_dias_scaler', 'rb' ) )
        df['prazo_em_dias'] = prazo_em_dias_scaler.transform(df[['prazo_em_dias']].values)

        dias_desde_cadastro_scaler = pickle.load( open( 'scalers/dias_desde_cadastro_scaler', 'rb' ) )
        df['dias_desde_cadastro'] = dias_desde_cadastro_scaler.transform(df[['dias_desde_cadastro']].values)

        len_credit_history_scaler = pickle.load( open( 'scalers/len_credit_history_scaler', 'rb' ) )
        df['len_credit_history'] = len_credit_history_scaler.transform(df[['len_credit_history']].values)

        ultima_data_emprestimo_scaler = pickle.load( open( 'scalers/ultima_data_emprestimo_scaler', 'rb' ) )
        df['ultima_data_emprestimo'] = ultima_data_emprestimo_scaler.transform(df[['ultima_data_emprestimo']].values)

        VALOR_A_PAGAR_scaler = pickle.load( open( 'scalers/VALOR_A_PAGAR_scaler', 'rb' ) )
        df['VALOR_A_PAGAR'] = VALOR_A_PAGAR_scaler.transform(df[['VALOR_A_PAGAR']].values)

        RENDA_MES_ANTERIOR_scaler = pickle.load( open( 'scalers/RENDA_MES_ANTERIOR_scaler', 'rb' ) )
        df['RENDA_MES_ANTERIOR'] = RENDA_MES_ANTERIOR_scaler.transform(df[['RENDA_MES_ANTERIOR']].values)

        valor_emprestimo_scaler = pickle.load( open( 'scalers/valor_emprestimo_scaler', 'rb' ) )
        df['valor_emprestimo'] = valor_emprestimo_scaler.transform(df[['valor_emprestimo']].values)

        diff_renda_scaler = pickle.load( open( 'scalers/diff_renda_scaler', 'rb' ) )
        df['diff_renda'] = diff_renda_scaler.transform(df[['diff_renda']].values)

        return df



    def make_prediction( self, df ):
        # Load trained model
        model = pickle.load( open( 'model/model.pkl' , 'rb' ) )

        # prediction
        pred = model.predict( df )
                
        # join pred into the original data
        df['INADIMPLENTE'] = pred
        df = df[['ID_CLIENTE','SAFRA_REF','INADIMPLENTE']]

        return df