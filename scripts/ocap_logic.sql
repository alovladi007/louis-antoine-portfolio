-- Out of Control Action Plan (OCAP) Logic
-- 193nm DUV Lithography Process Control
-- SQL Server Implementation for Academic Cleanroom MES

-- Create database schema for OCAP system
USE CleanroomDB;
GO

-- Create tables for OCAP tracking
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='OCAP_Actions' AND xtype='U')
CREATE TABLE OCAP_Actions (
    ActionID INT IDENTITY(1,1) PRIMARY KEY,
    WaferID VARCHAR(20) NOT NULL,
    Timestamp DATETIME DEFAULT GETDATE(),
    TriggerType VARCHAR(50) NOT NULL,
    TriggerValue FLOAT,
    ResponseLevel INT NOT NULL,
    ActionTaken VARCHAR(500),
    OperatorID VARCHAR(20),
    EngineerID VARCHAR(20),
    Status VARCHAR(20) DEFAULT 'OPEN',
    ResolutionTime DATETIME,
    Comments TEXT
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Hold_Table' AND xtype='U')
CREATE TABLE Hold_Table (
    HoldID INT IDENTITY(1,1) PRIMARY KEY,
    WaferID VARCHAR(20) NOT NULL,
    HoldType VARCHAR(50) NOT NULL,
    Reason VARCHAR(500) NOT NULL,
    Level INT NOT NULL,
    PlacedBy VARCHAR(20) NOT NULL,
    PlacedTime DATETIME DEFAULT GETDATE(),
    ReleasedBy VARCHAR(20),
    ReleasedTime DATETIME,
    Status VARCHAR(20) DEFAULT 'ACTIVE'
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Action_Required' AND xtype='U')
CREATE TABLE Action_Required (
    TaskID INT IDENTITY(1,1) PRIMARY KEY,
    WaferID VARCHAR(20) NOT NULL,
    Action VARCHAR(500) NOT NULL,
    Priority VARCHAR(20) DEFAULT 'MEDIUM',
    AssignedTo VARCHAR(20),
    CreatedTime DATETIME DEFAULT GETDATE(),
    DueTime DATETIME NOT NULL,
    CompletedTime DATETIME,
    Status VARCHAR(20) DEFAULT 'PENDING'
);

-- Main OCAP trigger procedure
CREATE OR ALTER PROCEDURE OCAP_ContactLitho
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @CD_Mean FLOAT, @CD_3Sigma FLOAT, @Bridge_Rate FLOAT, @Wafer_ID VARCHAR(20);
    DECLARE @Timestamp DATETIME = GETDATE();
    
    -- Get latest measurement
    SELECT TOP 1 
        @CD_Mean = CD_Mean,
        @CD_3Sigma = CD_3Sigma,
        @Bridge_Rate = Bridge_Rate,
        @Wafer_ID = Wafer_ID
    FROM Metrology_Data
    ORDER BY Timestamp DESC;
    
    -- Check for critical CD specification violations (Level 3)
    IF @CD_Mean < 85 OR @CD_Mean > 95
    BEGIN
        -- Level 3 response: Immediate process stop
        INSERT INTO OCAP_Actions (WaferID, TriggerType, TriggerValue, ResponseLevel, ActionTaken)
        VALUES (@Wafer_ID, 'CD_Critical_OOS', @CD_Mean, 3, 'Process stop initiated - Critical CD violation');
        
        INSERT INTO Hold_Table (WaferID, HoldType, Reason, Level, PlacedBy)
        VALUES (@Wafer_ID, 'PROCESS_STOP', 'Critical CD out of spec: ' + CAST(@CD_Mean AS VARCHAR), 3, 'SYSTEM');
        
        -- Send critical alert
        EXEC sp_SendCriticalAlert 'process_engineer@university.edu', 
             'CRITICAL: Process Stop Required', 
             @Wafer_ID, 
             'CD Mean: ' + CAST(@CD_Mean AS VARCHAR) + ' nm';
    END
    
    -- Check for CD specification violations (Level 2)
    ELSE IF @CD_Mean < 87 OR @CD_Mean > 93
    BEGIN
        -- Level 2 response: Engineering hold
        INSERT INTO OCAP_Actions (WaferID, TriggerType, TriggerValue, ResponseLevel, ActionTaken)
        VALUES (@Wafer_ID, 'CD_Mean_OOS', @CD_Mean, 2, 'Engineering hold placed - CD out of specification');
        
        INSERT INTO Hold_Table (WaferID, HoldType, Reason, Level, PlacedBy)
        VALUES (@Wafer_ID, 'ENGINEERING_HOLD', 'CD Mean OOS: ' + CAST(@CD_Mean AS VARCHAR), 2, 'SYSTEM');
        
        -- Send engineering notification
        EXEC sp_SendAlert 'litho_engineer@university.edu',
             'Engineering Hold: CD Mean out of spec',
             @Wafer_ID,
             'CD Mean: ' + CAST(@CD_Mean AS VARCHAR) + ' nm';
    END
    
    -- Check for high variation (Level 1)
    IF @CD_3Sigma > 3.0
    BEGIN
        -- Level 1 response: Operator verification
        INSERT INTO OCAP_Actions (WaferID, TriggerType, TriggerValue, ResponseLevel, ActionTaken)
        VALUES (@Wafer_ID, 'CD_High_Variation', @CD_3Sigma, 1, 'Operator verification required');
        
        INSERT INTO Action_Required (WaferID, Action, Priority, DueTime)
        VALUES (@Wafer_ID, 'Verify focus offset and re-measure CD uniformity', 'HIGH', 
                DATEADD(hour, 1, GETDATE()));
        
        -- Send operator notification
        EXEC sp_SendAlert 'litho_operator@university.edu',
             'Action Required: High CD variation detected',
             @Wafer_ID,
             'CD 3-sigma: ' + CAST(@CD_3Sigma AS VARCHAR) + ' nm';
    END
    
    -- Check for bridge defects (Level 2)
    IF @Bridge_Rate > 0.05
    BEGIN
        INSERT INTO OCAP_Actions (WaferID, TriggerType, TriggerValue, ResponseLevel, ActionTaken)
        VALUES (@Wafer_ID, 'Bridge_Rate_High', @Bridge_Rate, 2, 'Engineering review required - High bridge rate');
        
        INSERT INTO Hold_Table (WaferID, HoldType, Reason, Level, PlacedBy)
        VALUES (@Wafer_ID, 'QUALITY_HOLD', 'Bridge rate exceeded: ' + CAST(@Bridge_Rate AS VARCHAR), 2, 'SYSTEM');
        
        -- Send quality alert
        EXEC sp_SendAlert 'quality_engineer@university.edu',
             'Quality Hold: Bridge rate exceeded',
             @Wafer_ID,
             'Bridge Rate: ' + CAST(@Bridge_Rate AS VARCHAR) + ' defects/cm²';
    END
    
    -- Check for trending issues using statistical rules
    EXEC OCAP_CheckTrends @Wafer_ID;
    
END;
GO

-- Procedure to check for trending issues
CREATE OR ALTER PROCEDURE OCAP_CheckTrends
    @Wafer_ID VARCHAR(20)
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Get last 9 measurements for trend analysis
    DECLARE @RecentData TABLE (
        RowNum INT,
        CD_Mean FLOAT,
        Timestamp DATETIME
    );
    
    INSERT INTO @RecentData
    SELECT TOP 9 
        ROW_NUMBER() OVER (ORDER BY Timestamp DESC) as RowNum,
        CD_Mean,
        Timestamp
    FROM Metrology_Data
    WHERE WaferID LIKE LEFT(@Wafer_ID, 6) + '%'  -- Same lot
    ORDER BY Timestamp DESC;
    
    -- Rule 2: 9 points on same side of centerline
    DECLARE @Target FLOAT = 90.0;
    DECLARE @SameSideCount INT;
    
    SELECT @SameSideCount = COUNT(*)
    FROM @RecentData
    WHERE (CD_Mean > @Target AND RowNum <= 9)
       OR (CD_Mean < @Target AND RowNum <= 9);
    
    IF @SameSideCount = 9
    BEGIN
        INSERT INTO OCAP_Actions (WaferID, TriggerType, TriggerValue, ResponseLevel, ActionTaken)
        VALUES (@Wafer_ID, 'Trend_9_Same_Side', @SameSideCount, 1, 'Trend detected: 9 points same side of target');
        
        INSERT INTO Action_Required (WaferID, Action, Priority, DueTime)
        VALUES (@Wafer_ID, 'Investigate systematic shift in CD mean', 'MEDIUM', 
                DATEADD(hour, 4, GETDATE()));
    END
    
    -- Rule 5: 2 of 3 points beyond 2-sigma
    DECLARE @CD_StdDev FLOAT;
    SELECT @CD_StdDev = STDEV(CD_Mean) FROM @RecentData;
    
    DECLARE @Beyond2Sigma INT;
    SELECT @Beyond2Sigma = COUNT(*)
    FROM @RecentData
    WHERE RowNum <= 3 
      AND (ABS(CD_Mean - @Target) > 2 * @CD_StdDev);
    
    IF @Beyond2Sigma >= 2
    BEGIN
        INSERT INTO OCAP_Actions (WaferID, TriggerType, TriggerValue, ResponseLevel, ActionTaken)
        VALUES (@Wafer_ID, 'Trend_2of3_Beyond_2Sigma', @Beyond2Sigma, 1, 'Trend detected: 2 of 3 beyond 2-sigma');
        
        INSERT INTO Action_Required (WaferID, Action, Priority, DueTime)
        VALUES (@Wafer_ID, 'Check process stability and adjust if needed', 'HIGH', 
                DATEADD(hour, 2, GETDATE()));
    END
END;
GO

-- Procedure to handle OCAP responses
CREATE OR ALTER PROCEDURE OCAP_HandleResponse
    @ActionID INT,
    @ResponseAction VARCHAR(500),
    @OperatorID VARCHAR(20),
    @Comments TEXT = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    DECLARE @WaferID VARCHAR(20), @ResponseLevel INT;
    
    -- Get action details
    SELECT @WaferID = WaferID, @ResponseLevel = ResponseLevel
    FROM OCAP_Actions
    WHERE ActionID = @ActionID;
    
    -- Update action record
    UPDATE OCAP_Actions
    SET Status = 'IN_PROGRESS',
        ActionTaken = ActionTaken + ' | Response: ' + @ResponseAction,
        OperatorID = @OperatorID,
        Comments = @Comments
    WHERE ActionID = @ActionID;
    
    -- Log response action
    INSERT INTO Action_Required (WaferID, Action, AssignedTo, Priority, DueTime)
    VALUES (@WaferID, @ResponseAction, @OperatorID, 'HIGH', DATEADD(hour, 2, GETDATE()));
    
    -- Send confirmation
    EXEC sp_SendAlert @OperatorID + '@university.edu',
         'OCAP Response Logged',
         @WaferID,
         'Action: ' + @ResponseAction;
END;
GO

-- Procedure to close OCAP actions
CREATE OR ALTER PROCEDURE OCAP_CloseAction
    @ActionID INT,
    @Resolution VARCHAR(500),
    @EngineerID VARCHAR(20)
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Update action status
    UPDATE OCAP_Actions
    SET Status = 'CLOSED',
        ResolutionTime = GETDATE(),
        EngineerID = @EngineerID,
        Comments = ISNULL(Comments, '') + ' | Resolution: ' + @Resolution
    WHERE ActionID = @ActionID;
    
    -- Release any associated holds
    DECLARE @WaferID VARCHAR(20);
    SELECT @WaferID = WaferID FROM OCAP_Actions WHERE ActionID = @ActionID;
    
    UPDATE Hold_Table
    SET Status = 'RELEASED',
        ReleasedBy = @EngineerID,
        ReleasedTime = GETDATE()
    WHERE WaferID = @WaferID AND Status = 'ACTIVE';
    
    -- Complete any pending actions
    UPDATE Action_Required
    SET Status = 'COMPLETED',
        CompletedTime = GETDATE()
    WHERE WaferID = @WaferID AND Status = 'PENDING';
END;
GO

-- Alert procedures
CREATE OR ALTER PROCEDURE sp_SendAlert
    @Email VARCHAR(100),
    @Subject VARCHAR(200),
    @WaferID VARCHAR(20),
    @Message VARCHAR(1000)
AS
BEGIN
    -- In real implementation, would integrate with email system
    -- For demo, log to alerts table
    INSERT INTO Alert_Log (Email, Subject, WaferID, Message, SentTime)
    VALUES (@Email, @Subject, @WaferID, @Message, GETDATE());
    
    PRINT 'Alert sent to ' + @Email + ': ' + @Subject;
END;
GO

CREATE OR ALTER PROCEDURE sp_SendCriticalAlert
    @Email VARCHAR(100),
    @Subject VARCHAR(200),
    @WaferID VARCHAR(20),
    @Message VARCHAR(1000)
AS
BEGIN
    -- Critical alerts go to multiple recipients
    EXEC sp_SendAlert @Email, @Subject, @WaferID, @Message;
    EXEC sp_SendAlert 'fab_manager@university.edu', @Subject, @WaferID, @Message;
    EXEC sp_SendAlert 'safety_officer@university.edu', @Subject, @WaferID, @Message;
    
    -- Also log as critical
    INSERT INTO Critical_Alert_Log (Email, Subject, WaferID, Message, SentTime)
    VALUES (@Email, @Subject, @WaferID, @Message, GETDATE());
END;
GO

-- Create supporting tables for alerts
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Alert_Log' AND xtype='U')
CREATE TABLE Alert_Log (
    AlertID INT IDENTITY(1,1) PRIMARY KEY,
    Email VARCHAR(100) NOT NULL,
    Subject VARCHAR(200) NOT NULL,
    WaferID VARCHAR(20),
    Message VARCHAR(1000),
    SentTime DATETIME DEFAULT GETDATE(),
    Status VARCHAR(20) DEFAULT 'SENT'
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Critical_Alert_Log' AND xtype='U')
CREATE TABLE Critical_Alert_Log (
    CriticalAlertID INT IDENTITY(1,1) PRIMARY KEY,
    Email VARCHAR(100) NOT NULL,
    Subject VARCHAR(200) NOT NULL,
    WaferID VARCHAR(20),
    Message VARCHAR(1000),
    SentTime DATETIME DEFAULT GETDATE(),
    AcknowledgedBy VARCHAR(20),
    AcknowledgedTime DATETIME
);

-- Automated trigger to run OCAP on new measurements
CREATE OR ALTER TRIGGER tr_OCAP_NewMeasurement
ON Metrology_Data
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Run OCAP procedure for each new measurement
    DECLARE @WaferID VARCHAR(20);
    DECLARE measurement_cursor CURSOR FOR
    SELECT DISTINCT WaferID FROM inserted;
    
    OPEN measurement_cursor;
    FETCH NEXT FROM measurement_cursor INTO @WaferID;
    
    WHILE @@FETCH_STATUS = 0
    BEGIN
        EXEC OCAP_ContactLitho;
        FETCH NEXT FROM measurement_cursor INTO @WaferID;
    END
    
    CLOSE measurement_cursor;
    DEALLOCATE measurement_cursor;
END;
GO

-- Views for OCAP reporting
CREATE OR ALTER VIEW vw_OCAP_Dashboard
AS
SELECT 
    oa.ActionID,
    oa.WaferID,
    oa.Timestamp,
    oa.TriggerType,
    oa.TriggerValue,
    oa.ResponseLevel,
    CASE oa.ResponseLevel
        WHEN 1 THEN 'Operator Action'
        WHEN 2 THEN 'Engineering Hold'
        WHEN 3 THEN 'Process Stop'
    END as ResponseDescription,
    oa.Status,
    oa.OperatorID,
    oa.EngineerID,
    DATEDIFF(minute, oa.Timestamp, ISNULL(oa.ResolutionTime, GETDATE())) as ResponseTimeMinutes,
    ht.Status as HoldStatus,
    ar.Status as ActionStatus
FROM OCAP_Actions oa
LEFT JOIN Hold_Table ht ON oa.WaferID = ht.WaferID AND ht.Status = 'ACTIVE'
LEFT JOIN Action_Required ar ON oa.WaferID = ar.WaferID AND ar.Status = 'PENDING';
GO

-- Performance metrics view
CREATE OR ALTER VIEW vw_OCAP_Metrics
AS
SELECT 
    CAST(Timestamp as DATE) as Date,
    COUNT(*) as TotalActions,
    SUM(CASE WHEN ResponseLevel = 1 THEN 1 ELSE 0 END) as Level1Actions,
    SUM(CASE WHEN ResponseLevel = 2 THEN 1 ELSE 0 END) as Level2Actions,
    SUM(CASE WHEN ResponseLevel = 3 THEN 1 ELSE 0 END) as Level3Actions,
    AVG(CASE WHEN Status = 'CLOSED' 
        THEN DATEDIFF(minute, Timestamp, ResolutionTime) 
        ELSE NULL END) as AvgResolutionTimeMinutes,
    COUNT(CASE WHEN Status = 'OPEN' THEN 1 END) as OpenActions
FROM OCAP_Actions
WHERE Timestamp >= DATEADD(day, -30, GETDATE())
GROUP BY CAST(Timestamp as DATE);
GO

-- Example usage and testing
/*
-- Test OCAP system with sample data
INSERT INTO Metrology_Data (WaferID, CD_Mean, CD_3Sigma, Bridge_Rate, Timestamp)
VALUES ('W001-01', 85.5, 2.8, 0.03, GETDATE());  -- Should trigger Level 2

INSERT INTO Metrology_Data (WaferID, CD_Mean, CD_3Sigma, Bridge_Rate, Timestamp)
VALUES ('W001-02', 90.2, 3.5, 0.02, GETDATE());  -- Should trigger Level 1

INSERT INTO Metrology_Data (WaferID, CD_Mean, CD_3Sigma, Bridge_Rate, Timestamp)
VALUES ('W001-03', 84.0, 2.5, 0.08, GETDATE());  -- Should trigger Level 3

-- Check results
SELECT * FROM vw_OCAP_Dashboard ORDER BY Timestamp DESC;
SELECT * FROM Hold_Table WHERE Status = 'ACTIVE';
SELECT * FROM Action_Required WHERE Status = 'PENDING';
*/

PRINT 'OCAP system initialized successfully for 193nm lithography process';
