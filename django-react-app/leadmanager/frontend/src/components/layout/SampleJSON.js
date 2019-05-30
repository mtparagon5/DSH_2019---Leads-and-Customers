// rfc tab
import React, { Component, Fragment } from 'react'

export class SampleJSON extends Component {

  render() {

    const sampleLead = {
      "id": 0,
      "address": "27105 Christopher Station\r\nKevinborough, WA 91949-7197",
      "birthdate": "8/8/2010",
      "mail": "qturner@hotmail.com",
      "name": "Samuel Simpson",
      "sex": "M",
      "username": "debrapeterson",
      "days_since_signup": 18,
      "acquisition_channel": "Organic Search",
      "job_title": "Marketing Director",
      "company_size": "11 to 50",
      "completed_form": 1,
      "visited_pricing": 1,
      "registered_for_webinar": 0,
      "attended_webinar": 0,
      "industry": "Web & Internet",
      "converted": 1,
      "is_manager": false,
      "acquisition_channel_Cold_Call": 0,
      "acquisition_channel_Cold_Email": 0,
      "acquisition_channel_Organic_Search": 1,
      "acquisition_channel_Paid_Leads": 0,
      "acquisition_channel_Paid_Search": 0,
      "company_size_1_10": 0,
      "company_size_1000_10000": 0,
      "company_size_10001_plus": 0,
      "company_size_101_250": 0,
      "company_size_11_50": 1,
      "company_size_251_1000": 0,
      "company_size_51_100": 0,
      "industry_Financial_Services": 0,
      "industry_Furniture": 0,
      "industry_Heavy_Manufacturing": 0,
      "Scandanavion_Design": 0,
      "Transportation": 0,
      "Internet": 1,
      "score": 151
    };
    const jsonString = JSON.stringify(sampleLead, null, '\t');

    return (
      <div>
        <h1 className="display-4">JSON Data Structure</h1>
        <p className="lead">A SAMPLE VIEW OF THE DATA AVAILABLE</p>
        <textarea className="w-100" id="sample-json" style={{ height: 70 + 'vh' }} value={jsonString}></textarea>
      </div>
    )
  }
}

export default SampleJSON

