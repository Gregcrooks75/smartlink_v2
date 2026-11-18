import pyodbc
import pandas as pd

def get_transformer_data():
    # CMP query
    query_cmp = """
        WITH Transformer_without_structure AS (
            SELECT 
                x.SAPID,
                FeederID AS CircuitID,
                OperatingVoltage,
                SerialNumber,
                TransformerGUID,
                t.POLEID,
                CASE 
                    WHEN t.SubtypeCD = '2' THEN 'Single Phase Overhead'
                    WHEN t.SubtypeCD = '3' THEN 'Single Phase Underground'
                    WHEN t.SubtypeCD = '4' THEN 'Two Phase Open Delta'
                    WHEN t.SubtypeCD = '5' THEN 'Two Phase Open Wye'
                    WHEN t.SubtypeCD = '6' THEN 'Three Phase Overhead'
                    WHEN t.SubtypeCD = '7' THEN 'Three Phase Underground'
                    WHEN t.SubtypeCD = '8' THEN 'Step'
                END AS TransformerType,
                CASE 
                    WHEN x.Cocode = '9310' THEN 'CMP'
                END AS OPCO
            FROM MEGIS.Electric.TRANSFORMER_evw t
            INNER JOIN MEGIS.Electric.XFMRUNIT_evw x ON x.TransformerGUID = t.GlobalID
        )
        SELECT * FROM Transformer_without_structure
    """

    # NYSEG/RGE query
    query_ny = """
        WITH Transformer_without_structure AS (
            SELECT 
                x.SAPID,
                FeederID AS CircuitID,
                OperatingVoltage,
                SerialNumber,
                TransformerGUID,
                t.POLEID,
                CASE 
                    WHEN t.SubtypeCD = '2' THEN 'Single Phase Overhead'
                    WHEN t.SubtypeCD = '3' THEN 'Single Phase Underground'
                    WHEN t.SubtypeCD = '4' THEN 'Two Phase Open Delta'
                    WHEN t.SubtypeCD = '5' THEN 'Two Phase Open Wye'
                    WHEN t.SubtypeCD = '6' THEN 'Three Phase Overhead'
                    WHEN t.SubtypeCD = '7' THEN 'Three Phase Underground'
                    WHEN t.SubtypeCD = '8' THEN 'Step'
                END AS TransformerType,
                CASE 
                    WHEN x.Cocode = '9301' THEN 'NYSEG'
                    WHEN x.Cocode = '9302' THEN 'RGE'
                END AS OPCO
            FROM NYGIS.Electric.TRANSFORMER_evw t
            INNER JOIN NYGIS.Electric.XFMRUNIT_evw x ON x.TransformerGUID = t.GlobalID
        )
        SELECT * FROM Transformer_without_structure
    """

    # Database connection strings
    conn_str_cmp = (
        "Driver={SQL Server};"
        "Server=cmpgisprd19;"
        "Database=MEGIS;"
        "Trusted_Connection=yes;"
    )

    conn_str_ny = (
        "Driver={SQL Server};"
        "Server=nygisprd19;"
        "Database=NYGIS;"
        "Trusted_Connection=yes;"
    )

    # Fetch data
    with pyodbc.connect(conn_str_cmp) as conn1:
        df_cmp = pd.read_sql_query(query_cmp, conn1)

    with pyodbc.connect(conn_str_ny) as conn2:
        df_ny = pd.read_sql_query(query_ny, conn2)

    # Combine both
    df_all = pd.concat([df_cmp, df_ny], ignore_index=True)
    return df_all

# Optional: Test block
if __name__ == "__main__":
    df_transformers = get_transformer_data()
