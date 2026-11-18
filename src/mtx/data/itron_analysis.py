import pandas as pd
from mtx.db.db_AMI import run_query_with_temp_ami
# NY Query
server = "ct-omsdbprod.osg.avangrid.net"
ny_database = "NYAMI_SDE"
query_ny = """


-- Getting Itron Analysis Data
DROP TABLE IF EXISTS #TA_TABLE
SELECT Meter_Number, Tx_Subgroup_ID, LA_Transformer_ID, GIS_Transformer_ID, Nearest_Transformer_ID INTO #TA_TABLE
FROM [NYAMI_SDE].[NYAMI].[TA_METER]
	
-- Counting Subgroups by Transformer
DROP TABLE IF EXISTS #Subgroup_Count_By_Transformer
SELECT GIS_Transformer_ID, COUNT(DISTINCT Tx_Subgroup_ID) AS Subgroup_Count INTO #Subgroup_Count_By_Transformer
FROM #TA_TABLE
GROUP BY GIS_Transformer_ID


-- Extracting the Transformer Load table in order to identify subgroup discrepancies
DROP TABLE IF EXISTS #Transformer_Load
SELECT Transformer_ID, OnePhase_3Phase INTO #Transformer_Load
FROM [NYAMI].[TA_Transformer_LOAD]


SELECT * FROM #Transformer_Load
-- Get The most up to date GIS Transformer IDs as well as the Superior FLOCs. For instances where there are 
DROP TABLE IF EXISTS #Updated_GIS_Transformer_ID;
WITH RankedMeters AS (
    SELECT 
        meterid, 
        txid,
		SuperiorFLOC,
        ROW_NUMBER() OVER (PARTITION BY meterid ORDER BY txid DESC) AS rn
    FROM vw_current_sapCustomer_TXs_onPole
)

SELECT 
    meterid, 
    txid,
	SuperiorFLOC
INTO #Updated_GIS_Transformer_ID
FROM RankedMeters
WHERE rn = 1;





-- Making Sure that the TA Table has the updated GIS Transformer IDs

DROP TABLE IF EXISTS #Updated_TA_Table
SELECT Meter_Number, Tx_Subgroup_ID, LA_Transformer_ID, COALESCE(txid, GIS_Transformer_ID) AS GIS_Transformer_ID, Nearest_Transformer_ID, COALESCE(n.SuperiorFLOC, vw.SuperiorFLOC) AS SuperiorFLOC
INTO #Updated_TA_Table
FROM #TA_TABLE a 
LEFT JOIN #Updated_GIS_Transformer_ID vw 
ON vw.meterid = a.Meter_Number
LEFT JOIN NYAMI.ALL_METERS_NY n ON n.MeterID = a.Meter_Number



-- Finding Count of Subgroups By FLOC
DROP TABLE IF EXISTS #Subgroup_Count_by_FLOC
SELECT SuperiorFLOC, Count(DISTINCT Tx_Subgroup_ID) AS Unique_Subgroups_FLOC  INTO #Subgroup_Count_by_FLOC
FROM #Updated_TA_Table GROUP BY SuperiorFLOC
SELECT * FROM #Subgroup_Count_by_FLOC



-- Adding Subgroup Count Analysis to table
DROP TABLE IF EXISTS #TA_TABLE_Subgroup_FLOCS
SELECT Meter_Number, Tx_Subgroup_ID, LA_Transformer_ID, GIS_Transformer_ID, Nearest_Transformer_ID, s.SuperiorFLOC, Unique_Subgroups_FLOC  INTO #TA_TABLE_Subgroup_FLOCS
FROM #Updated_TA_Table u 
LEFT JOIN #Subgroup_Count_by_FLOC s ON s.SuperiorFLOC = u.SuperiorFLOC



-- Check Itron Recommendation, Nearest, and GIS_Transformer_Match 
DROP TABLE IF EXISTS #Discrepancy_Check2
SELECT Meter_Number, Tx_Subgroup_ID, LA_Transformer_ID, GIS_Transformer_ID, Nearest_Transformer_ID,
CASE
    WHEN LA_Transformer_ID = GIS_Transformer_ID AND LA_Transformer_ID = Nearest_Transformer_ID 
        THEN 'Accurate: All Transformers Match'
    WHEN LA_Transformer_ID = Nearest_Transformer_ID AND GIS_Transformer_ID != LA_Transformer_ID 
        THEN 'Discrepancy: Itron Xfmr = Nearest Transformer AND  Nearest Transformer != GIS Transformer ID'
    WHEN LA_Transformer_ID = GIS_Transformer_ID AND LA_Transformer_ID != Nearest_Transformer_ID 
        THEN 'Discrepancy: Itron Xfmr = GIS Transformer ID AND  Nearest Transformer != GIS Transformer ID'
    WHEN GIS_Transformer_ID = Nearest_Transformer_ID AND LA_Transformer_ID != Nearest_Transformer_ID OR GIS_Transformer_ID = Nearest_Transformer_ID AND LA_Transformer_ID IS NULL
        THEN 'Accurate: GIS Transformer ID = Nearest Transformer'
	WHEN GIS_Transformer_ID != Nearest_Transformer_ID AND LA_Transformer_ID != Nearest_Transformer_ID 
        THEN 'Discrepancy: None of the Transformers Match'
	WHEN GIS_Transformer_ID IS NULL THEN 'Discrepancy: GIS Transformer is Null'
	WHEN GIS_Transformer_ID != Nearest_Transformer_ID AND LA_Transformer_ID IS NULL THEN 'Discrepancy: GIS Transformer is not the nearest Transformer'
    ELSE 'Discrepancy'
END AS Xfmr_Discrepancy
INTO #Discrepancy_Check2
FROM #Updated_TA_Table



-- Joining the table containing the Transformers phases to see if the phase count matches the Subgroup Count for the given transformer
DROP TABLE IF EXISTS #Final_Analysis
SELECT 
    f.Meter_Number,
	Tx_Subgroup_ID,
    f.GIS_Transformer_ID,
    f.LA_Transformer_ID,
    f.Nearest_Transformer_ID,
    f.Unique_Subgroups_FLOC,
    CASE 
        WHEN f.Unique_Subgroups_FLOC <= t.OnePhase_3Phase 
        THEN 'Accurate: Subgroups count <= Transformer count in FLOC' 
        ELSE 'Discrepancy: More Subgroups than Transformers in FLOC' 
    END AS Subgroup_Discrepancy,
	SuperiorFLOC
INTO #Final_Analysis  
FROM #TA_TABLE_Subgroup_FLOCS f 
LEFT JOIN #Transformer_Load t ON t.Transformer_ID = f.GIS_Transformer_ID


-- Create the temp table using SELECT INTO
-- Drop the temp table if it already exists
DROP TABLE IF EXISTS #PhaseSummary;
-- Define CTEs
WITH PhaseCounts AS (
    SELECT 
        Z_POLEID, 
        CASE 
            WHEN PHASE IN ('A', 'B', 'C') THEN 1
            WHEN PHASE IN ('AB', 'AC', 'BC') THEN 2
            WHEN PHASE IN ('ABC', 'CBA') THEN 3
            ELSE 0
        END AS NumPhases
    FROM DXFR
),
TransformerCounts AS (
    SELECT 
        Z_POLEID, 
        COUNT(SAPID) AS Transformer_Count_Pole
    FROM DXFR
    WHERE SAPID IS NOT NULL
    GROUP BY Z_POLEID
)

-- Combine both into a temp table
SELECT 
    pc.Z_POLEID,
    SUM(pc.NumPhases) AS TotalPhases
INTO #PhaseSummary
FROM PhaseCounts pc
LEFT JOIN TransformerCounts tc ON pc.Z_POLEID = tc.Z_POLEID
GROUP BY pc.Z_POLEID, tc.Transformer_Count_Pole;


DROP TABLE IF EXISTS #PoleIDSummary
SELECT SAPID, CASE WHEN COCODE = '9301' THEN 'NYSEG' ELSE 'RGE' END AS OPCO,Z_GISID, p.Z_POLEID, p.TotalPhases, TOTCUST 
INTO #PoleIDSummary
FROM DXFR d
LEFT JOIN #PhaseSummary p ON p.Z_POLEID = d.Z_POLEID

-- Drop temp table if it exists
DROP TABLE IF EXISTS #FinalPoleTransformerSummary;

-- Create the temp table with combined data
SELECT 
    f.Meter_Number,
    f.Tx_Subgroup_ID,
    f.GIS_Transformer_ID,
    f.LA_Transformer_ID,
    f.Nearest_Transformer_ID,
    f.SuperiorFLOC,
    p.OPCO,
    p.SAPID,
    p.Z_GISID,
    p.Z_POLEID,
	p.TOTCUST AS CustomersCountTransformer,
	f.Unique_Subgroups_FLOC,
    p.TotalPhases AS TotalPhasesInPole,
	CASE WHEN
		TotalPhases >= f.Unique_Subgroups_FLOC THEN 'Accurate: Phases Equal or Superior to Subgroups'
		ELSE 'Discrepancy: More Subgroups than Phases'
	END AS PhaseSubgroup_Discrepancy_Check
		
INTO #FinalPoleTransformerSummary
FROM #Final_Analysis f
LEFT JOIN #PoleIDSummary p 
    ON p.Z_GISID = f.GIS_Transformer_ID


DROP TABLE IF EXISTS #RearrangedTable;
SELECT f.*, u.Xfmr_Discrepancy INTO #RearrangedTable
FROM #FinalPoleTransformerSummary f
LEFT JOIN #Discrepancy_Check2 u
    ON u.Meter_Number = f.Meter_Number 

DROP TABLE IF EXISTS #TableWithConfidencePercentage
SELECT
    f.Meter_Number,
    f.Tx_Subgroup_ID,
    f.GIS_Transformer_ID,
    f.LA_Transformer_ID,
    f.Nearest_Transformer_ID,
    f.Unique_Subgroups_FLOC,
    f.Subgroup_Discrepancy,
    f.SuperiorFLOC,
    p.OPCO,
    p.TOTCUST AS CustomersCountTransformer,
    p.SAPID,
    p.Z_GISID,
    p.Z_POLEID,
    p.TotalPhases AS TotalPhasesInPole,

    /* ---------------- Association Confidence % (last column) ---------------- */
    CASE 
        /* --- All three match --- */
        WHEN f.LA_Transformer_ID IS NOT NULL 
             AND f.LA_Transformer_ID = f.GIS_Transformer_ID 
             AND f.LA_Transformer_ID = f.Nearest_Transformer_ID THEN
            CASE 
                WHEN p.TotalPhases >= f.Unique_Subgroups_FLOC THEN 100
                WHEN p.TotalPhases < f.Unique_Subgroups_FLOC THEN 90
                ELSE 100 -- if phase info missing, default to the ≥ branch
            END

        /* --- GIS = Nearest (LA differs or NULL) --- */
        WHEN f.GIS_Transformer_ID IS NOT NULL 
             AND f.GIS_Transformer_ID = f.Nearest_Transformer_ID 
             AND (f.LA_Transformer_ID IS NULL OR f.LA_Transformer_ID <> f.Nearest_Transformer_ID) THEN
            CASE 
                WHEN p.TotalPhases >= f.Unique_Subgroups_FLOC THEN 90
                WHEN p.TotalPhases < f.Unique_Subgroups_FLOC THEN 80
                ELSE 90
            END

        /* --- GIS = LA (Nearest differs or NULL) --- */
        WHEN f.GIS_Transformer_ID IS NOT NULL 
             AND f.LA_Transformer_ID IS NOT NULL 
             AND f.GIS_Transformer_ID = f.LA_Transformer_ID 
             AND (f.Nearest_Transformer_ID IS NULL OR f.Nearest_Transformer_ID <> f.GIS_Transformer_ID) THEN
            CASE 
                WHEN p.TotalPhases >= f.Unique_Subgroups_FLOC THEN 60
                WHEN p.TotalPhases < f.Unique_Subgroups_FLOC THEN 50
                ELSE 60
            END

        /* --- LA = Nearest (GIS differs or NULL) --- */
        WHEN f.LA_Transformer_ID IS NOT NULL 
             AND f.Nearest_Transformer_ID IS NOT NULL 
             AND f.LA_Transformer_ID = f.Nearest_Transformer_ID 
             AND (f.GIS_Transformer_ID IS NULL OR f.GIS_Transformer_ID <> f.Nearest_Transformer_ID) THEN
            CASE 
                WHEN p.TotalPhases >= f.Unique_Subgroups_FLOC THEN 30
                WHEN p.TotalPhases < f.Unique_Subgroups_FLOC THEN 20
                ELSE 30
            END

        /* --- None match --- */
        ELSE
            CASE 
                WHEN p.TotalPhases >= f.Unique_Subgroups_FLOC THEN 10
                WHEN p.TotalPhases < f.Unique_Subgroups_FLOC THEN 0
                ELSE 10
            END
    END AS GISAssociationConfidencePct
	INTO #TableWithConfidencePercentage
FROM 
    #Final_Analysis f
LEFT JOIN 
    #PoleIDSummary p ON p.Z_GISID = f.GIS_Transformer_ID;



"""
final_query = """SELECT * FROM #TableWithConfidencePercentage"""


def ami_data_from_db():
    df_ny = run_query_with_temp_ami(query_ny, final_query, server, ny_database)
    return df_ny




































