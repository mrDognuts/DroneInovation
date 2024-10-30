
import { type } from 'os';
// Import the PostgreSQL library

const { Client } = require('pg');     // Get client information from PGAdmin4

// Create a new client instance with your database connection details
const client = new Client({
  user: 'EYECON-PC',                // pc name
  host: 'localhost',                // database host
  database: 'InsuranceDB',          // database name
  password: 'EMMANUEL7',            // password for accessing database on PostgreSQL    
  port: 5432,                       // default PostgreSQL port
});

// Connect to the database
client.connect()
  .then(() => {
    console.log('Database connected successfully');
    
    // Return the current time from the database

    return client.query('SELECT NOW()');
  })
  .then((res) => {
    console.log('Database response:', res.rows[0]);
  })
  .catch((err) => {
    console.error('There was an error establishing a connection to the database', err.stack);
  })
  .finally(() => {
    // Close the database connection
    client.end();
  });

  
