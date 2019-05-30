import React, { Component, Fragment } from 'react';
import ReactDOM from 'react-dom';

import { Header } from './layout/Header';
import { SampleJSON } from './layout/SampleJSON';
import Dashboard from './leads/Dashboard';

import { Provider } from 'react-redux';
import store from '../store';

// import 'react-virtualized/styles.css'; // only needs to be imported once

class App extends Component {
    render() {
        return (
            <Provider store={store}>
                <Fragment>
                    <Header />
                    <div className="container">
                        <Dashboard />
                        <SampleJSON />
                    </div>
                </Fragment>
            </Provider>
        )
    }
}


ReactDOM.render(<App />, document.getElementById('app'));