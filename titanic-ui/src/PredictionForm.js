import React, { useState } from 'react';

function PredictionForm() {
    const [formData, setFormData] = useState({
        Pclass: 3,
        Sex: "male",
        Age: 22,
        Fare: 7.25,
        Embarked: "S",
        SibSp: 0,
        Parch: 0
    });

    const [result, setResult] = useState(null);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value
        });
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                ...formData,
                Pclass: Number(formData.Pclass),
                Age: Number(formData.Age),
                Fare: Number(formData.Fare),
                SibSp: Number(formData.SibSp),
                Parch: Number(formData.Parch)
            })
        });

        const data = await response.json();
        setResult(data);
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <label>
                    Class:
                    <input name="Pclass" value={formData.Pclass} onChange={handleChange} />
                </label>
                <br />

                <label>
                    Sex:
                    <select name="Sex" value={formData.Sex} onChange={handleChange}>
                        <option value="male">Male</option>
                        <option value="female">Female</option>
                    </select>
                </label>
                <br />

                <label>
                    Age:
                    <input name="Age" value={formData.Age} onChange={handleChange} />
                </label>
                <br />

                <label>
                    Fare:
                    <input name="Fare" value={formData.Fare} onChange={handleChange} />
                </label>
                <br />

                <label>
                    Embarked:
                    <select name="Embarked" value={formData.Embarked} onChange={handleChange}>
                        <option value="S">S</option>
                        <option value="C">C</option>
                        <option value="Q">Q</option>
                    </select>
                </label>
                <br />

                <label>
                    Siblings/Spouse:
                    <input name="SibSp" value={formData.SibSp} onChange={handleChange} />
                </label>
                <br />

                <label>
                    Parents/Children:
                    <input name="Parch" value={formData.Parch} onChange={handleChange} />
                </label>
                <br />

                <button type="submit">Predict</button>
            </form>

            {result && (
                <div style={{ marginTop: "20px" }}>
                    <h3>Result</h3>
                    <p>Survived: {result.survived === 1 ? "Yes" : "No"}</p>
                    <p>Probability: {result.survival_probability}</p>
                </div>
            )}
        </div>
    );
}

export default PredictionForm;
