const cds = require('@sap/cds');
const express = require('express');

cds.on('bootstrap', app => {
  app.use(express.json());
  // Add any other middleware or routes here
});

module.exports = cds.server;
