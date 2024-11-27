const axios = require('axios');

const token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiNjczZWUxNzlhMmQ3N2VmNGMzYjQ4OTNiIiwicm9sZSI6ImFkbWluIn0sImlhdCI6MTczMjE3NDIwOSwiZXhwIjoxNzMyMTc3ODA5fQ.5oHhh8h3ykfczX6awf0eFLYu2AKVyRuQW3_PdjynxeY';

const simulateVehicle = async () => {
    try {
        // Get available parking spots
        const spotsResponse = await axios.get('http://localhost:3001/api/parkingspots', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        // Find an available spot
        const availableSpot = spotsResponse.data.find(spot => spot.status === 'available' && !spot.occupied);
        
        if (!availableSpot) {
            throw new Error('No available parking spots');
        }

        // Simulate vehicle entry
        const vehicleData = {
            licensePlate: 'ABC123',
            make: 'Toyota',
            model: 'Camry',
            color: 'Blue',
            spotId: availableSpot.spotId
        };

        const response = await axios.post('http://localhost:3001/api/vehicles/enter', vehicleData, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        console.log('Vehicle entry simulated:', response.data);
    } catch (error) {
        console.error('Error simulating vehicle entry:', error.response ? error.response.data : error.message);
    }
};

simulateVehicle();
