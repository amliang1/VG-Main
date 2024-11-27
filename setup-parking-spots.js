const axios = require('axios');

const token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiNjczZWUxNzlhMmQ3N2VmNGMzYjQ4OTNiIiwicm9sZSI6ImFkbWluIn0sImlhdCI6MTczMjE3NDIwOSwiZXhwIjoxNzMyMTc3ODA5fQ.5oHhh8h3ykfczX6awf0eFLYu2AKVyRuQW3_PdjynxeY';

const setupParkingSpots = async () => {
    try {
        // Create test parking spots
        const spots = [
            {
                spotId: 'A1',
                type: 'standard',
                section: 'A',
                status: 'available',
                occupied: false,
                location: {
                    level: 1,
                    building: 'Main'
                }
            },
            {
                spotId: 'A2',
                type: 'electric',
                section: 'A',
                status: 'available',
                occupied: false,
                location: {
                    level: 1,
                    building: 'Main'
                }
            }
        ];

        for (const spot of spots) {
            const response = await axios.post('http://localhost:3001/api/parkingspots', spot, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            console.log(`Created parking spot ${spot.spotId}:`, response.data);
        }
    } catch (error) {
        console.error('Error setting up parking spots:', error.response ? error.response.data : error.message);
    }
};

setupParkingSpots();
