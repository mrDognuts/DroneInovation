
-- DROP DATABASE IF EXISTS "InsuranceDB";

CREATE SEQUENCE clientkeys
START WITH 625340;  				   -- Define the starting value for ClientID's as 625340


-- Table for Client information
CREATE TABLE tblClients
(
	ClientID 		INT DEFAULT nextval('clientkeys') 			NOT NULL,
	First_Name 		VARCHAR										NOT NULL,
	Last_Name  		VARCHAR										NOT NULL,
	DOB 	   		DATE										NOT NULL,
	Gender			CHAR(1)										NOT NULL,
	Email			VARCHAR										NOT NULL,
	PRIMARY KEY 	(ClientID)          					-- Set ClientID as Primary Key[PK]
);

-- Table for storing user login details

CREATE TABLE tblCredentials
(
	UserName		VARCHAR						NOT NULL,
	Password		VARCHAR						NOT NULL,
	PRIMARY KEY    (UserName)								-- Set Username as Primary KEY[PK]
 
);

-- Bridge table for Client table and Login information

CREATE TABLE tblClientCredentials
(
    ClientID INT,
    Username VARCHAR,
    PRIMARY KEY (ClientID, Username),						-- Set ClientID & Username as PK
    FOREIGN KEY (ClientID) REFERENCES tblClients(ClientID),
    FOREIGN KEY (Username) REFERENCES tblCredentials(Username)
);

-- Create default values for drone maintenance
CREATE TYPE droneStatus AS ENUM ('Active', 'In Maintenance');

CREATE TABLE tblDrones
(
   DroneID 					SERIAL				NOT NULL,
   Model					VARCHAR(30)			NOT NULL,
   SerialNo					SERIAL				NOT NULL,
   Status					droneStatus		    NOT NULL,
   Last_Maintenance			DATE				NOT NULL,
   Battery_Life				INT					NOT NULL,
   DateAdded				DATE				NOT NULL,		-- Date drone was added to the system
   PRIMARY KEY  			(DroneID)							-- Set DroneID as PK
);

-- Create table for Claims
CREATE TABLE tblClaims
(
	ReferenceNo				SERIAL				NOT NULL,		
	Amount					MONEY				NOT NULL,
	DamageType				VARCHAR				NOT NULL,
	IncidentDate			DATE				NOT NULL,
	PoliceReport			BYTEA				NOT NULL,		-- Image documentation of police incident report
	DroneID					INT					NOT NULL,		-- Foreign key for drone associated with claim
	PRIMARY KEY 			(ReferenceNo),
	FOREIGN KEY    			(DroneID)			REFERENCES		tblDrones(DroneID)
		
);

-- Join table for Clients & Claims
CREATE TABLE tblClientClaims
(
	ReferenceNo				INT					NOT NULL,
	ClientID				INT					NOT NULL,
	PRIMARY KEY 			(ReferenceNo,ClientID),
	FOREIGN KEY 			(ReferenceNo)					REFERENCES 		 tblClaims(ReferenceNo),
	FOREIGN KEY 			(ClientID)						REFERENCES 		 tblClients(ClientID)
);

-- create default values for flight status
CREATE TYPE flightStatus AS ENUM ('Completed', 'In-Progress', 'Aborted');

--create table for flights
CREATE TABLE tblFlights
(
   FlightID					SERIAL				NOT NULL,
   DroneID					INT					NOT NULL,
   ClientID					INT					NOT NULL,
   StartTime				TIME				NOT NULL,
   EndTime					TIME				NOT NULL,
   StartLocation			VARCHAR(50)			NOT NULL,
   EndLocation				VARCHAR(50)			NOT NULL,
   Status					flightStatus		NOT NULL,
   Notes					VARCHAR(60)			NOT NULL,			-- description of flight, i.e. weather
   PRIMARY KEY 				(FlightID) 								-- set flight id as PK
);

-- join table for drones and flights
CREATE TABLE tblDroneFlights
(
	DroneID					 INT				NOT NULL,
	FlightID				 INT				NOT NULL,
	PRIMARY KEY  			(DroneID,FlightID),
	FOREIGN KEY    			(DroneID)			REFERENCES		tblDrones(DroneID),
	FOREIGN KEY    			(FlightID)			REFERENCES		tblFlights(FlightID)   
);

-- join table for clients and flights
CREATE TABLE tblClientFlights
(
	ClientID				INT					NOT NULL,
	FlightID				INT					NOT NULL,
	PRIMARY KEY  			(ClientID,FlightID),
	FOREIGN KEY    			(ClientID)			REFERENCES		tblClients(ClientID),
	FOREIGN KEY    			(FlightID)			REFERENCES		tblFlights(FlightID) 
);

 -- table for images
CREATE TABLE tblImages
(   
	ImageID					SERIAL				NOT NULL,
	ImageUrl				VARCHAR(20)			NOT NULL,			
	CaptureTime				TIME				NOT NULL,
	GPS_Coordinates			VARCHAR(50)			NOT NULL,
	Image_Description		VARCHAR(60)			NOT NULL,
	PRIMARY KEY 			(ImageID)							-- set image id as PK

);

-- create default values for maintenance description
CREATE TYPE mDescription AS ENUM ('Routine Check','Battery Replacement','Firmware Update','Propeller Inspection');

CREATE TABLE tblMaintenanceLogs
(
   mID						SERIAL					NOT NULL,
   mDate					DATE					NOT NULL,
   performedBy				VARCHAR(30)				NOT NULL,		-- name of person who did the maintenance
   mType					mDescription			NOT NULL,
   Notes					VARCHAR(60)				NOT NULL,
   PRIMARY KEY 				(mID)
);

-- join table for drones and maintenance
CREATE TABLE tblDroneMaintenance
(
	DroneID						INT						NOT NULL,
	mID							INT						NOT NULL,
	PRIMARY KEY (DroneID,mID),
	FOREIGN KEY (DroneID)		REFERENCES				tblDrones(DroneID),
	FOREIGN KEY (mID)			REFERENCES				tblMaintenanceLogs(mID)
);

-- default values for compliance type and compliance result type
CREATE TYPE cType AS ENUM ('No Fly Zone','Licensing','Data Protection','Safety Protocols','Consent');
CREATE TYPE resType AS ENUM ('Failed','Passed');

CREATE TABLE tblCompliance
(
   cID							SERIAL 					NOT NULL,
   checkDate					DATE					NOT NULL,
   regulationType				cType					NOT NULL,	
   cResult						resType					NOT NULL,
   details						VARCHAR(60)				NOT NULL,
   PRIMARY KEY (cID)												-- set compliance id as PK
);


CREATE TABLE tblFlightCompliance
(
	FlightID						INT						NOT NULL,
	cID								INT						NOT NULL,
	PRIMARY KEY (FlightID,cID),
	FOREIGN KEY (FlightID)			REFERENCES tblFlights(FlightID),
	FOREIGN KEY (cID)				REFERENCES tblCompliance(cID)
);

INSERT INTO tblClients (First_Name, Last_Name, DOB, Gender, Email)
VALUES 
    ('Alice', 'Smith', '1990-04-23', 'F', 'alice.smith@example.com'),
    ('Bob', 'Johnson', '1985-11-12', 'M', 'bob.johnson@example.com'),
    ('Carol', 'Williams', '1992-07-08', 'F', 'carol.williams@example.com'),
    ('David', 'Brown', '1988-02-19', 'M', 'david.brown@example.com'),
    ('Eve', 'Davis', '1995-09-30', 'F', 'eve.davis@example.com');
	
INSERT INTO tblCredentials (UserName, Password)
VALUES 
    ('alice.smith', 'password123'),
    ('bob.johnson', 'securePass456'),
    ('carol.williams', 'pass789word'),
    ('david.brown', 'myPassword101'),
    ('eve.davis', 'passwordQWERTY');

INSERT INTO tblClientCredentials (ClientID, Username)
VALUES 
    (625345, 'alice.smith'),
    (625346, 'bob.johnson'),
    (625347, 'carol.williams'),
    (625348, 'david.brown'),
    (625349, 'eve.davis');


