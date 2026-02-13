// Suppress DEP0060 deprecation warning from dependencies
process.noDeprecation = true;

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const connectDB = require('./src/config/db');

const app = express();
connectDB();

app.use(cors());
app.use(express.json());

// serve uploaded images
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// public routes
app.use('/api/uploads', require('./src/routes/upload'));
app.use('/api/detect', require('./src/routes/detect'));
app.use('/api/auth', require('./src/routes/auth'));

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => console.log(`Backend running on ${PORT}`));
