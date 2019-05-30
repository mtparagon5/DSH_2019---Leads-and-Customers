import React, { Component } from 'react';


export class Header extends Component {
  render() {
    return (
      <div className="navbar navbar-expand-sm fixed-top navbar-dark bg-primary">
        <div className="container">
          <a href="#" className="navbar-brand">Lead Manager</a>
          <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarResponsive">
            <ul className="navbar-nav">
              <li className="nav-item dropdown">
                <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" id="themes">Data Structure<span className="caret"></span></a>
                <div className="dropdown-menu" aria-labelledby="themes">
                  <a className="dropdown-item" href="">Sample JSON</a>
                  <div className="dropdown-divider"></div>
                  <a className="dropdown-item" target="_blank" href="https://www.w3schools.com/whatis/whatis_json.asp">About JSON</a>
                  <div className="dropdown-divider"></div>
                  <a className="dropdown-item" href="#" >Download CSV</a>
                  <a className="dropdown-item" href="#" >Download JSON</a>
                </div>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="">Customers</a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="">Leads</a>
              </li>
              <li className="nav-item dropdown">
                <a className="nav-link dropdown-toggle" data-toggle="dropdown" href="#" id="download">Analysis <span className="caret"></span></a>
                <div className="dropdown-menu" aria-labelledby="download">
                  <a className="dropdown-item" target="_blank" href="">Data Visualization</a>
                  <div className="dropdown-divider"></div>
                  <a className="dropdown-item" href="" >Charts</a>
                  <a className="dropdown-item" href="" >Table Format</a>
                </div>
              </li>
            </ul>
          </div>
        </div>
      </div>
    )

  }
}