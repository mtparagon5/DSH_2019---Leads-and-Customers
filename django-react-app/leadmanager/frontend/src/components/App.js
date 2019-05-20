import React, {Component, Fragment} from 'react';
import ReactDOM from 'react-dom';

import { Header } from './layout/Header';
import { SampleJSON } from './layout/SampleJSON';

class App extends Component {
    render() {
        return (
            <Fragment>
                <Header/>
                <div className="container">
                    <SampleJSON/>
                </div>
            </Fragment>
        )
    }
}


ReactDOM.render(<App/>, document.getElementById('app'));