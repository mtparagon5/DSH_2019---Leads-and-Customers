import React, { Component, Fragment } from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import { getLeads } from '../../actions/leads';

import { Column, Table } from 'react-virtualized';


export class Leads extends Component {
    static propTypes = {
        // leads: PropTypes.array.isRequired
        leads: PropTypes.array.isRequired
    }

    componentDidMount() {
        this.props.getLeads();
    }


    render() {
        // const list = [{ name: 'allie', description: 'test description' }]
        // const test = 2.3;
        // const _leads = this.props.leads;
        // console.log(_leads);
        return (
            // <Fragment>
            // {/* <h2>Leads</h2> */ }
            < Table
                width={1000}
                height={500}
                headerHeight={50}
                rowHeight={80}
                rowCount={this.props.leads.length}
                rowGetter={({ index }) => this.props.leads[index]
                }
            >
                <Column
                    label='Id'
                    dataKey='id'
                    width={50}
                />
                <Column
                    label='Name'
                    dataKey='name'
                    width={250}
                />
                <Column
                    label='Days Since Sign Up'
                    dataKey='days_since_signup'
                    width={250}
                />
                <Column
                    label='Registered for Webinar'
                    dataKey='registered_for_webinar'
                    width={250}
                />
                <Column
                    width={200}
                    label='Converted'
                    dataKey='converted'
                />
            </Table >
            /* <table className="table table-responsive table-striped table-hover">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Days Since Sign Up</th>
                    <th>Registered for Webinar</th>
                    <th>Converted</th>
                </tr>

            </thead>
            <tbody>
                {this.props.leads[0].map(lead => (
                    <tr key={lead.id}>
                        <td>{lead.id}</td>
                        <td>{lead.name}</td>
                        <td>{lead.days_since_signup}</td>
                        <td>{lead.registered_for_webinar}</td>
                        <td>{lead.converted}</td>
                    </tr>
                ))}
            </tbody>
        </table> */
            // </Fragment>
        )
    }
}

const mapStateToProps = state => ({
    leads: state.leads.leads
})

export default connect(mapStateToProps, { getLeads })(Leads);
